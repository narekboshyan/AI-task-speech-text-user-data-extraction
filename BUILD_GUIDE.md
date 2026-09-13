# Build Guide — step by step, with code

Companion to `CHALLENGE_GUIDE.md` (which explains *what* and *why*). This one is *how*.
Follow it top to bottom. Each step ends with something you can run.

---

## The stack, and why

| Layer | Choice | Why |
|---|---|---|
| Speech-to-text | **`faster-whisper` large-v3, local** | Free, no API signup, strong German, and it accepts an `initial_prompt` you can prime with domain vocabulary. 14 min of audio runs in ~2–3 min on your M4. |
| Extraction | **Claude `claude-opus-5`** via the `anthropic` SDK | Structured outputs guarantee schema-valid JSON. Needs one API key. |
| Glue | Python 3.11 + `uv` | Already on your machine. |

Alternatives worth naming in your README (this is the "model choice" discussion they want):
**Deepgram `nova-3`** — hosted, faster, and supports *keyterm boosting* (you can literally
boost `gmx.de`, `Bindestrich`, `Unterstrich`), which is arguably a better fit but needs a key.
**`mlx-whisper`** — same model, Metal-accelerated for Apple Silicon, noticeably faster than
`faster-whisper` on an M4; swap it in if transcription feels slow.

> Heads up: `large-v3` downloads ~3 GB on first run. You have ~11 GB free — fine, but don't
> also pull a second model "just to compare" without checking disk.

---

## Step 0 — Scaffold (10 min)

```bash
cd ~/Downloads/phonebot_challenge
mkdir -p src/phonebot prompts tests out/transcripts
touch src/phonebot/__init__.py
```

`pyproject.toml`:

```toml
[project]
name = "phonebot"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40",
    "faster-whisper>=1.0",
    "pydantic>=2",
    "pyyaml>=6",
    "python-dotenv>=1",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/phonebot"]
```

```bash
uv venv && uv pip install -e ".[dev]"
echo "ANTHROPIC_API_KEY=" > .env.example
cp .env.example .env      # paste your real key into .env
printf '.env\n.venv/\nout/\n__pycache__/\n' > .gitignore
```

`config.yaml`:

```yaml
paths:
  recordings: data/recordings
  ground_truth: data/ground_truth.json
  transcripts: out/transcripts
  predictions: out/predictions.json

asr:
  model: large-v3
  language: de
  compute_type: int8

llm:
  model: claude-opus-5
  max_tokens: 4000
  prompt_version: v2      # <- the A/B knob
```

**Checkpoint:** `uv run python -c "import anthropic, faster_whisper; print('ok')"`

---

## Step 1 — Transcribe, and cache (40 min)

The `initial_prompt` is the important line here. Whisper conditions its decoder on it, so
seeding it with the vocabulary of spelled-out German emails measurably improves how it
renders letters, domains and separators.

`src/phonebot/transcribe.py`:

```python
"""Speech-to-text with an on-disk cache. Never transcribe the same file twice."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

# Primes the decoder toward spelled-out German contact details.
# Whisper conditions on this text, so it biases letter/domain/separator rendering.
ASR_PROMPT = (
    "Telefonat auf Deutsch. Der Anrufer buchstabiert seine E-Mail-Adresse und "
    "seine Telefonnummer. Vokabular: at, ätt, Klammeraffe, Punkt, Bindestrich, "
    "Minus, Unterstrich, Schrägstrich, Doppelpunkt, groß, klein, "
    "A wie Anton, B wie Berta, D wie Dora, F wie Friedrich, H wie Heinrich, "
    "M wie Martha, N wie Nordpol, S wie Samuel, T wie Theodor, Z wie Zeppelin. "
    "Domains: gmail.com, web.de, gmx.de, gmx.net, outlook.com, outlook.de, "
    "outlook.fr, t-online.de, yahoo.de, yahoo.com, yahoo.fr, freenet.de, "
    "libero.it, hotmail.es, uol.com.br, icloud.com."
)


@lru_cache(maxsize=1)
def _model(name: str, compute_type: str) -> WhisperModel:
    # Loading large-v3 takes ~20s; do it once per process.
    return WhisperModel(name, device="cpu", compute_type=compute_type)


def transcribe(wav: Path, cache_dir: Path, *, model: str, language: str,
               compute_type: str, force: bool = False) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{wav.stem}.json"
    if cached.exists() and not force:
        return json.loads(cached.read_text())

    segments, info = _model(model, compute_type).transcribe(
        str(wav),
        language=language,
        initial_prompt=ASR_PROMPT,
        word_timestamps=True,       # per-word confidence -> your monitoring signal later
        vad_filter=True,            # drops leading/trailing silence
    )

    segs = [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text.strip(),
            "avg_logprob": s.avg_logprob,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "prob": w.probability}
                for w in (s.words or [])
            ],
        }
        for s in segments          # NOTE: this generator is lazy — iterating it runs the model
    ]

    result = {
        "id": wav.stem,
        "text": " ".join(s["text"] for s in segs),
        "language": info.language,
        "duration": info.duration,
        "segments": segs,
    }
    cached.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
```

Run it over everything once:

```bash
uv run python -c "
from pathlib import Path
from phonebot.transcribe import transcribe
for w in sorted(Path('data/recordings').glob('*.wav')):
    r = transcribe(w, Path('out/transcripts'), model='large-v3', language='de', compute_type='int8')
    print(r['id'], '|', r['text'][:110])
"
```

**Now stop and actually read 3–4 transcripts.** Everything you build next is a response to
what you see in them. You'll notice things like `ätt` vs `at`, `H wie Heinrich`, digits
written as words, `zwo` instead of `zwei`.

**Checkpoint:** 30 files in `out/transcripts/`.

---

## Step 2 — The eval harness (45 min) ← DO THIS BEFORE ANY PROMPT WORK

This is the step people skip, and it's why they waste hours. Write the scorer against a
dummy extractor that returns nulls. Score 0/120. Now every change you make has a number.

`src/phonebot/eval.py`:

```python
"""Score predictions against ground truth. Reports exact and normalized accuracy."""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

FIELDS = ["first_name", "last_name", "email", "phone_number"]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def normalize(field: str, value: str | None) -> str:
    """Canonical form for comparison. Documented, deliberate, and the same both sides."""
    if value is None:
        return ""
    v = value.strip()
    if field == "phone_number":
        # "+49 152 11223456" and "+4915211223456" are the same number.
        digits = re.sub(r"\D", "", v)
        return "+" + digits.lstrip("0") if v.lstrip().startswith("+") else digits
    if field == "email":
        return v.lower()
    # Names: collapse whitespace; treat "Lisa-Marie" == "Lisa Marie" (per the brief's example).
    return re.sub(r"\s+", " ", v.replace("-", " ")).strip().casefold()


def matches(field: str, predicted: str | None, expected) -> bool:
    """Ground truth may be a string OR a list of acceptable values."""
    accepted = expected if isinstance(expected, list) else [expected]
    return any(normalize(field, predicted) == normalize(field, a) for a in accepted)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein. Tells you whether a miss is a prompt bug (1-2) or an ASR miss (5+)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(predictions: dict[str, dict], truth_path: Path) -> str:
    truth = json.loads(truth_path.read_text())["recordings"]
    tally = {f: {"exact": 0, "norm": 0} for f in FIELDS}
    failures, full_records = [], 0

    for rec in truth:
        pred = predictions.get(rec["id"], {})
        all_ok = True
        for f in FIELDS:
            exp, got = rec["expected"][f], pred.get(f)
            first_exp = exp[0] if isinstance(exp, list) else exp
            if got == first_exp or (isinstance(exp, list) and got in exp):
                tally[f]["exact"] += 1
            if matches(f, got, exp):
                tally[f]["norm"] += 1
            else:
                all_ok = False
                failures.append(
                    f"  {rec['id']}  {f:<13} got {str(got)!r:<34} "
                    f"want {first_exp!r:<34} (dist {edit_distance(str(got), str(first_exp))})"
                )
        full_records += all_ok

    n = len(truth)
    lines = [f"{'field':<15}{'exact':>9}{'norm':>9}{'exact%':>9}{'norm%':>9}", "-" * 51]
    for f in FIELDS:
        e, m = tally[f]["exact"], tally[f]["norm"]
        lines.append(f"{f:<15}{e:>6}/{n:<2}{m:>6}/{n:<2}{100*e/n:>8.1f}{100*m/n:>9.1f}")
    te = sum(t["exact"] for t in tally.values())
    tm = sum(t["norm"] for t in tally.values())
    tot = n * len(FIELDS)
    lines += [
        "-" * 51,
        f"{'OVERALL':<15}{te:>5}/{tot:<3}{tm:>5}/{tot:<3}{100*te/tot:>8.1f}{100*tm/tot:>9.1f}",
        f"\nfull-record exact (4/4): {full_records}/{n} ({100*full_records/n:.1f}%)",
    ]
    if failures:
        lines += ["\nfailures:"] + failures
    return "\n".join(lines)


if __name__ == "__main__":
    preds = {r["id"]: r.get("extracted", {}) for r in
             json.loads(Path("out/predictions.json").read_text())}
    report = score(preds, Path("data/ground_truth.json"))
    print(report)
    Path("out/scores.md").write_text(f"```\n{report}\n```\n")
```

**Checkpoint:** write `out/predictions.json` by hand as `[{"id":"call_01","extracted":{}}]`,
run `uv run python -m phonebot.eval`, see a table of zeros. Now you have a ruler.

---

## Step 3 — Normalization (45 min) ← the highest-value code in the project

Deterministic, free, instant, unit-testable. This is the German-speech-to-symbols layer.

`src/phonebot/normalize.py`:

```python
"""Turn spoken German contact details into symbols and digits.

Deterministic and unit-tested. Runs BEFORE the LLM. We pass both the raw and the
normalized transcript to the model — the model is often better at ambiguity, but it
needs the original evidence to work from.
"""
from __future__ import annotations

import re

# German spelling alphabet ("H wie Heinrich" -> "H")
SPELLING = {
    "anton": "A", "ärger": "Ä", "berta": "B", "cäsar": "C", "charlotte": "C",
    "dora": "D", "emil": "E", "friedrich": "F", "gustav": "G", "heinrich": "H",
    "ida": "I", "julius": "J", "kaufmann": "K", "konrad": "K", "ludwig": "L",
    "martha": "M", "nordpol": "N", "otto": "O", "ökonom": "Ö", "paula": "P",
    "quelle": "Q", "richard": "R", "samuel": "S", "siegfried": "S",
    "theodor": "T", "ulrich": "U", "übermut": "Ü", "viktor": "V",
    "wilhelm": "W", "xanthippe": "X", "ypsilon": "Y", "zacharias": "Z",
    "zeppelin": "Z",
}

DIGITS = {
    "null": "0", "eins": "1", "ein": "1", "zwei": "2", "zwo": "2", "drei": "3",
    "vier": "4", "fünf": "5", "fuenf": "5", "sechs": "6", "sieben": "7",
    "acht": "8", "neun": "9",
}

SYMBOLS = {
    "klammeraffe": "@", "ätt": "@", "ett": "@", "at-zeichen": "@",
    "punkt": ".", "dot": ".",
    "bindestrich": "-", "minus": "-", "trennstrich": "-", "strich": "-",
    "unterstrich": "_", "underscore": "_", "unterstreichen": "_",
    "schrägstrich": "/", "slash": "/",
}


def normalize(text: str) -> str:
    """Rewrite spoken forms as symbols/digits. Order matters."""
    t = text

    # 1. "X wie Anton" -> "X"   (do this first; it consumes words that look like names)
    t = re.sub(
        r"\b([A-Za-zÄÖÜäöü])\s+wie\s+(\w+)",
        lambda m: SPELLING.get(m.group(2).lower(), m.group(1).upper()),
        t,
    )

    # 2. "doppel s" / "doppel-l" -> "ss"
    t = re.sub(r"\bdoppel[\s-]*([a-zäöü])\b", lambda m: m.group(1) * 2, t, flags=re.I)

    # 3. standalone "at" -> "@"  (guarded: "at" is rare as a German word, but be explicit)
    t = re.sub(r"(?<=\s)at(?=\s)", "@", t, flags=re.I)

    # 4. symbols and digits, longest-key-first so "at-zeichen" beats "at"
    for table in (SYMBOLS, DIGITS):
        for word in sorted(table, key=len, reverse=True):
            t = re.sub(rf"\b{re.escape(word)}\b", table[word], t, flags=re.I)

    return re.sub(r"\s{2,}", " ", t)


def to_email_local(name: str) -> str:
    """Umlaut/accent rules for the EMAIL field only. Names keep their diacritics."""
    import unicodedata
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        name = name.lower().replace(a, b)
    return "".join(c for c in unicodedata.normalize("NFD", name)
                   if unicodedata.category(c) != "Mn")
```

**Write the tests now** — this is the cheapest test-coverage win in the challenge:

`tests/test_normalize.py`:

```python
from phonebot.normalize import normalize, to_email_local


def test_spelling_alphabet():
    assert normalize("H wie Heinrich vier sieben") == "H 4 7"

def test_symbols():
    assert normalize("web punkt de") == "web . de"
    assert normalize("Bindestrich") == "-"
    assert normalize("Unterstrich") == "_"

def test_underscore_not_confused_with_hyphen():
    assert "_" in normalize("michael Unterstrich wagner")
    assert "-" not in normalize("michael Unterstrich wagner")

def test_zwo_is_two():
    assert normalize("null eins fünf zwo") == "0 1 5 2"

def test_email_transliteration_but_names_keep_diacritics():
    assert to_email_local("Schröder") == "schroeder"
    assert to_email_local("García") == "garcia"
    assert to_email_local("O'Brien") == "o'brien"
```

```bash
uv run pytest -q
```

---

## Step 4 — Extraction (45 min)

`prompts/extract_v2.md` — the prompt lives in a **file**, not a string literal, so you can
version and A/B it without touching code:

```markdown
You extract caller contact details from German AI-phonebot call transcripts.

You receive the raw transcript and a pre-normalized version where spoken German symbols
have been rewritten (at/ätt → @, Punkt → ., Bindestrich → -, Unterstrich → _,
"H wie Heinrich" → H, number words → digits). Use both. The raw transcript is the
evidence; the normalized one is a hint that may be wrong.

## German spelling conventions
- "at" / "ätt" / "Klammeraffe" = @
- "Punkt" = .          "Bindestrich" / "Minus" / "Strich" = -
- "Unterstrich" = _    "Schrägstrich" = /
- "X wie Anton" is the German spelling alphabet: it means the single letter X.
- "zwo" = 2 (used on phones to avoid confusion with "drei").
- "doppel-s" = ss

## Field rules — read carefully, these differ per field
- first_name / last_name: write them as they are actually spelled, KEEPING all
  diacritics and apostrophes. "Schröder", "García", "Martínez", "O'Brien".
- email: lowercase, and transliterate/strip diacritics the way email addresses do —
  ä→ae, ö→oe, ü→ue, ß→ss, and accents dropped. So the caller "Julia Schröder" with
  address "julia punkt schroeder ätt web punkt de" gives julia.schroeder@web.de.
  Never put an umlaut or accent in an email address.
- phone_number: German format "+49 <area> <rest>". Mobile area codes are 3 digits
  (151, 160, 176...). Landline area codes are 2 or 3 digits (30 Berlin, 89 Munich,
  40 Hamburg, 69 Frankfurt, 221 Cologne, 511 Hannover). A leading spoken "null" after
  the country code is dropped. Transcribe the digits EXACTLY as spoken — do not
  shorten, pad, or "fix" a number that looks unusual.

## Behaviour
- If the caller corrects themselves, the LAST stated value wins. Note it in `notes`.
- If a field was never given, return null. NEVER invent or guess a plausible value —
  a wrong value silently corrupts a CRM record, a null gets a human's attention.
- For each field, quote the exact transcript span you took it from in `evidence`.
- Give an honest 0.0–1.0 confidence per field.
```

`src/phonebot/extract.py`:

```python
"""One LLM call per recording. Structured output, so the JSON is always schema-valid."""
from __future__ import annotations

import random
import time
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from .normalize import normalize


class Confidence(BaseModel):
    first_name: float
    last_name: float
    email: float
    phone_number: float


class Extraction(BaseModel):
    first_name: str | None = Field(description="Keeps diacritics, e.g. Schröder")
    last_name: str | None
    email: str | None = Field(description="Lowercase, ASCII only, umlauts transliterated")
    phone_number: str | None = Field(description='German format "+49 152 11223456"')
    confidence: Confidence
    evidence: dict[str, str] = Field(description="Transcript span each value came from")
    notes: str | None = None


client = anthropic.Anthropic()


def _with_retries(fn, attempts: int = 4):
    """Retry only what is retryable: 429 and 5xx. Never retry a 400."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except anthropic.RateLimitError as e:
            last = e
        except anthropic.APIStatusError as e:
            if e.status_code < 500:
                raise                       # our bug — fail loudly, don't hammer the API
            last = e
        except anthropic.APIConnectionError as e:
            last = e
        time.sleep(min(2 ** i + random.random(), 30))
    raise last


def extract(transcript: dict, *, model: str, prompt_path: Path, max_tokens: int) -> Extraction:
    system = prompt_path.read_text()
    raw = transcript["text"]
    user = (
        f"RAW TRANSCRIPT:\n{raw}\n\n"
        f"PRE-NORMALIZED:\n{normalize(raw)}\n\n"
        "Extract the caller's contact details."
    )
    response = _with_retries(lambda: client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=Extraction,
    ))
    return response.parsed_output
```

> `client.messages.parse(..., output_format=<pydantic model>)` returns `.parsed_output`
> already validated. That's why "model returned broken JSON" is not a failure mode you
> have to hand-code around — mention this in your README as a deliberate choice.

---

## Step 5 — Validation + confidence (30 min)

`src/phonebot/validate.py`:

```python
"""Validate, flag, and route for review. Deliberately does NOT overwrite the model."""
from __future__ import annotations

import re

from .normalize import to_email_local

EMAIL_RE = re.compile(r"^[a-z0-9][a-z0-9._%+-]*@[a-z0-9.-]+\.[a-z]{2,}$")

KNOWN_DOMAINS = {
    "gmail.com", "web.de", "gmx.de", "gmx.net", "outlook.com", "outlook.de",
    "outlook.fr", "t-online.de", "yahoo.de", "yahoo.com", "yahoo.fr",
    "freenet.de", "libero.it", "hotmail.es", "uol.com.br", "icloud.com",
}


def _dist(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def validate(rec: dict) -> dict:
    """Attach warnings + needs_human_review. Never silently rewrites a value."""
    warnings: list[str] = []
    email, phone = rec.get("email"), rec.get("phone_number")

    if email:
        if not EMAIL_RE.match(email):
            warnings.append("email_failed_regex")
        if any(c in email for c in "äöüßáíéó"):
            warnings.append("email_contains_diacritic")   # should have been transliterated
        domain = email.rsplit("@", 1)[-1]
        if domain not in KNOWN_DOMAINS:
            near = [d for d in KNOWN_DOMAINS if _dist(domain, d) <= 2]
            warnings.append(f"unknown_domain:{domain}" + (f" near={near}" if near else ""))

    if phone:
        digits = re.sub(r"\D", "", phone)
        if not digits.startswith("49"):
            warnings.append("phone_missing_country_code")
        # Length check FLAGS only. call_02's real answer is 12 digits after +49 —
        # "correcting" unusual-but-real numbers would destroy correct answers.
        if not (10 <= len(digits) <= 16):
            warnings.append(f"phone_unusual_length:{len(digits)}")

    # Cross-check: the email local part usually encodes the name.
    if email and rec.get("last_name"):
        local = email.split("@")[0]
        if to_email_local(rec["last_name"]) not in local and \
           to_email_local(rec["last_name"])[:4] not in local:
            warnings.append("name_email_mismatch")

    conf = rec.get("confidence") or {}
    lowest = min(conf.values()) if conf else 0.0

    rec["warnings"] = warnings
    rec["needs_human_review"] = bool(warnings) or lowest < 0.7
    return rec
```

---

## Step 6 — Wire it together (30 min)

`src/phonebot/pipeline.py`:

```python
"""Run the whole thing. One bad recording must never kill the batch."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .extract import extract
from .transcribe import transcribe
from .validate import validate


def main() -> int:
    load_dotenv()
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    p, asr, llm = cfg["paths"], cfg["asr"], cfg["llm"]
    prompt = Path(f"prompts/extract_{llm['prompt_version']}.md")

    results, failed = [], 0
    for wav in sorted(Path(p["recordings"]).glob("*.wav")):
        t0 = time.perf_counter()
        row = {"id": wav.stem, "prompt_version": llm["prompt_version"], "model": llm["model"]}
        try:
            tr = transcribe(wav, Path(p["transcripts"]), model=asr["model"],
                            language=asr["language"], compute_type=asr["compute_type"])

            if tr["duration"] < 1.0 or not tr["text"].strip():
                raise ValueError("silent_or_empty_audio")

            data = extract(tr, model=llm["model"], prompt_path=prompt,
                           max_tokens=llm["max_tokens"]).model_dump()
            row["extracted"] = validate(data)
            row["transcript"] = tr["text"]
        except Exception as e:                      # isolate: the other 29 must still run
            failed += 1
            row["extracted"] = {f: None for f in
                                ("first_name", "last_name", "email", "phone_number")}
            row["error"] = f"{type(e).__name__}: {e}"

        row["latency_s"] = round(time.perf_counter() - t0, 2)
        results.append(row)
        flag = "!" if "error" in row else ("?" if row["extracted"].get("needs_human_review") else " ")
        print(f"{flag} {row['id']}  {row['extracted'].get('email')}", file=sys.stderr)

    Path(p["predictions"]).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n{len(results)} processed, {failed} failed -> {p['predictions']}", file=sys.stderr)
    return 1 if failed else 0                       # non-zero exit if anything broke


if __name__ == "__main__":
    raise SystemExit(main())
```

**Run the full loop:**

```bash
uv run python -m phonebot.pipeline
uv run python -m phonebot.eval
```

Write down the number. That's your v1 baseline.

---

## Step 7 — Iterate with the scorer (60 min, the actual work)

Now the loop is: read `failures:` in the eval output → find the pattern → fix the *right*
layer → re-score.

Use the edit distance to decide which layer:

| Edit distance | Diagnosis | Where to fix |
|---|---|---|
| 1–2 | The model heard it, formatted it wrong | Prompt rule (`prompts/extract_v3.md`) |
| 1–2, and it's a separator (`-` vs `_`) | Spoken-symbol confusion | `normalize.py` + a prompt line |
| 3+ on an email | ASR never got the letters | `ASR_PROMPT` vocabulary, or a better ASR |
| Right value, wrong shape | Your normalizer in `eval.py` is too strict | `eval.normalize()` — but document the decision |

To A/B a prompt, change one line in `config.yaml` and re-run:

```bash
sed -i '' 's/prompt_version: v2/prompt_version: v3/' config.yaml
uv run python -m phonebot.pipeline && uv run python -m phonebot.eval
```

Transcripts are cached, so a re-run is LLM-only — about 30 seconds and a few cents.
**Record each variant's score in a table.** That table going into your README *is* your
A/B testing story, and it's evidence rather than a promise.

---

## Step 8 — README (30 min) — weighted heavily, don't rush it

Structure it like this:

```markdown
# Phonebot extraction pipeline

## Run it
    uv sync && cp .env.example .env   # add ANTHROPIC_API_KEY
    uv run python -m phonebot.pipeline    # ~4 min first run (downloads Whisper large-v3)
    uv run python -m phonebot.eval

## Results
<paste out/scores.md>

## What moved the number
| variant                          | email  | overall |
|----------------------------------|--------|---------|
| v1: transcript -> LLM            |  58.3% |   80.8% |
| v2: + German symbol normalization|  83.3% |   92.5% |
| v3: + ASR vocabulary priming     |  90.0% |   94.2% |

## Design decisions
- Whisper large-v3 locally, primed with an `initial_prompt` of German spelling
  vocabulary and the known email domains — [+X pts on email].
- Symbol normalization in deterministic Python, not in the prompt: it's free, instant,
  and unit-tested. The LLM gets both raw and normalized text.
- Diacritics are handled per-field: names keep ö/í/', emails transliterate. This is the
  single rule that costs the most accuracy if you get it wrong.
- Validate, don't correct. `call_02`'s phone number is 12 digits — longer than a real
  German mobile — so a strict validator would have "fixed" a correct answer. Validation
  lowers confidence and flags for review; it never overwrites.
- Structured outputs (Pydantic) instead of JSON parsing, so malformed JSON isn't a
  failure mode at all.

## Failure handling
<the table from CHALLENGE_GUIDE §6>

## Production: monitoring, extensibility
<§7 of CHALLENGE_GUIDE, in your own words>

## What I'd do next
- ASR n-best candidates re-ranked by the LLM against the email grammar
- Second audio-native model as a cross-check; disagreement -> human review
- Active learning from human corrections; bounced emails are free labels
```

---

## Order of operations, compressed

```
scaffold → transcribe+cache → EVAL HARNESS → normalize+tests → prompt v1 → score
  → read failures → fix layer → score → repeat → validate/confidence → README → Loom
```

**If you run out of time, cut** the confidence scoring, the cross-check, the A/B variants.
**Never cut** the eval harness, the results table, or per-record error isolation.
