# JUPUS Phonebot Challenge — Engineer's Guide

> Written for someone who has never done a speech→structured-data pipeline before.
> Read top to bottom once, then use Section 9 as your build checklist.

---

## 1. What they actually want, in one paragraph

JUPUS runs AI phone bots. The bot calls (or is called by) a person, has a short German
conversation, and asks for their name, email and phone number. The recording lands in a
bucket. Somebody now has to turn `call_17.wav` into:

```json
{ "first_name": "Sophie", "last_name": "Dubois",
  "email": "sophie.dubois@outlook.fr", "phone_number": "+49 171 66234567" }
```

Your job: build the thing that does that automatically, and **prove how well it does it**
by scoring against the 30 labelled calls they gave you.

That's it. Audio in → four fields out → a number that says how accurate you are.

### What they're really grading

Read the recruiter email again. Five criteria, and only one is "accuracy":

| Criterion | What a reviewer looks for |
|---|---|
| Extraction accuracy | A number. Per-field and overall. Measured by *your* eval script, not by vibes. |
| Pipeline / AI approach | Did you pick transcription + prompting deliberately, and can you say why? |
| Edge cases & failures | What happens when the API 500s, the audio is silent, the model returns garbage JSON? |
| Code quality | Can a stranger run it in 2 minutes and read it in 10? |
| Monitoring & future work | Can you talk about production without hand-waving? |

Note what is **not** on the list: a web UI, a database, Docker, Kubernetes, a queue.
The email says literally *"You do not need to over-engineer the solution."* Budget is
2–5 hours. Build the spine well; describe the rest in the README.

---

## 2. What's actually in the box

```
phonebot_challenge/
├── README.md                     ← the official brief
└── data/
    ├── ground_truth.json         ← 30 records, 4 fields each = 120 labels
    └── recordings/
        └── call_01.wav … call_30.wav
```

Facts I measured from the files:

- **30 recordings**, WAV, PCM 16-bit, **mono, 44.1 kHz**
- **18.9 s – 39.0 s** each, median 28.9 s, **14.3 minutes total**
- All German speech
- Total audio is tiny. Transcribing everything costs cents and takes a couple of minutes.
  **You can afford to re-run the whole pipeline dozens of times.** Design for that.

⚠️ **Trap:** the example in their `README.md` (`Jürgen Meyer`, `+4917284492`) does **not**
appear in the real `ground_truth.json`. It's illustrative. Read the real file.

⚠️ **Second trap:** the brief says some values may be *arrays* of acceptable answers
(`"first_name": ["Lisa Marie", "Lisa-Marie"]`). In the 30 files you were given, **none are
arrays** — they're all plain strings. Your scorer must still handle both, because they may
run it against a held-out set that does have them. This is a free "handled the edge case"
point; it takes four lines:

```python
def matches(predicted: str | None, expected: str | list[str]) -> bool:
    accepted = expected if isinstance(expected, list) else [expected]
    return any(normalize(predicted) == normalize(a) for a in accepted)
```

---

## 3. The part that will actually decide your score

Naive plan: "transcribe the audio, ask an LLM for the fields." That gets you maybe 95% on
names and **40–60% on emails**. Here's why, and where every point is won or lost.

### 3.1 Names are easy. Emails and phone numbers are not.

Names are spoken as words, and speech models are good at words. Emails and phone numbers
are **spelled out and read digit-by-digit**, in German, for 10–20 seconds. That's where the
errors live.

A caller saying `h47-herbst@web.de` produces German audio roughly like:

> "h wie Heinrich, vier, sieben, Bindestrich, h-e-r-b-s-t, at, w-e-b, Punkt, d-e"

Your speech-to-text model will hand you something like:

> `H wie Heinrich vier sieben Bindestrich Herbst ätt web punkt de`

Now you need to know that:

| German in the transcript | Means |
|---|---|
| `at`, `ätt`, `Klammeraffe` | `@` |
| `Punkt` | `.` |
| `Bindestrich`, `Minus`, `Strich` | `-` |
| `Unterstrich`, `Underscore` | `_` |
| `A wie Anton`, `H wie Heinrich` | the letter `A`, the letter `H` (German spelling alphabet) |
| `null eins fünf zwo` | `0 1 5 2` (`zwo` is spoken `zwei`, used on phones to avoid confusion with `drei`) |
| `doppel-fünf`, `zweimal die drei` | `55`, `33` |
| `ä ö ü` in an email | almost always written `ae oe ue` |

**This mapping is the core engineering problem of the challenge.** Everything else is
plumbing.

### 3.2 Look at the real ground truth — the edge cases are deliberate

I went through all 30 labels. They are clearly hand-designed to break naive pipelines:

**Emails — every special character appears at least once:**

| Call | Email | What it's testing |
|---|---|---|
| 02 | `h47-herbst@web.de` | digits inside the local part + hyphen |
| 05 | `sandra-weber@t-online.de` | hyphen in local part **and in the domain** |
| 06 | `k.fischer1983@gmail.com` | a year read as digits or as "neunzehnhundertdreiundachtzig" |
| 09 | `petra-m@yahoo.de` | single-letter segment |
| 12 | `michael_wagner@gmx.de` | **underscore** — easy to confuse with hyphen |
| 22 | `lucas.silva@uol.com.br` | two-level TLD; regex must not stop at `.com` |

16 distinct domains appear: `gmail.com, web.de, gmx.de, gmx.net, outlook.com, outlook.de,
outlook.fr, t-online.de, yahoo.de, yahoo.com, yahoo.fr, freenet.de, libero.it, hotmail.es,
uol.com.br, icloud.com`. Note `gmx.de` **and** `gmx.net`, `outlook.com` **and** `outlook.de`
— so you cannot just snap every "gmx" to one answer. A domain allowlist helps, but it must
be a *tie-breaker over plausible candidates*, not a blind overwrite.

**Diacritics — the single nastiest rule in the whole dataset:**

| Call | Name (keeps the accent) | Email (drops/transliterates it) |
|---|---|---|
| 07 | Julia **Schröder** | julia.**schroeder**@web.de |
| 09 | Petra **Müller** | petra-m@yahoo.de |
| 20 | Carlos **García** | carlos.**garcia**@hotmail.es |
| 29 | Isabella **Martínez** | isabella.**martinez**@gmail.com |
| 30 | Liam **O'Brien** | liam.**obrien**@gmail.com |

So: **`last_name` must preserve `ö/í/'`, `email` must not.** If you write one prompt that
says "normalize special characters," you lose the name fields. If you say "keep everything
exactly as spoken," you lose the email fields. You need to state the rule per field.

**Names — half the dataset isn't German:**
calls 01–15 are German (Schmidt, Hoffmann, Zimmermann, Schröder…), calls 16–30 are not
(Anderson, Dubois, Rossi, García, Kowalski, Silva, Andersson, Chen, Lefevre, Hassan, Smith,
Tanaka, Martínez, O'Brien). A German-only speech model will happily turn *Tanaka* into
*Tanacka* and *Kowalski* into *Kowalsky*. This is testing whether your transcription config
survives foreign names inside German speech.

**Phone numbers — two different formats, and one that shouldn't validate:**

- 24 mobile numbers: `+49 1xx xxxxxxxx` (`151, 152, 157, 159, 160, 162, 163, 169, 170…179`)
- 6 landlines: `+49 30` Berlin, `+49 89` Munich, `+49 40` Hamburg, `+49 221` Cologne,
  `+49 69` Frankfurt, `+49 511` Hannover — **area codes are 2 or 3 digits, not fixed width**
- `call_02` is `+49 157 231412313` — that's 12 digits after the country code, which is
  **longer than a real German mobile number**.

That last one matters a lot. If you pipe everything through a strict validator like the
`phonenumbers` library and "correct" invalid numbers, you will *destroy* a correct answer.

> **Rule: validate, don't silently correct.** Use validation to lower a confidence score and
> flag for review. Never let a validator overwrite what was actually said.

### 3.3 The scoring arithmetic

30 calls × 4 fields = **120 cells**. Each cell is 0.83% of the overall score.
Per-field, each call is 3.3%.

That means: **one systematically-fixed email bug is worth up to 25 points.** Getting the
email normalization right is worth more than anything else you could spend the time on.

---

## 4. The architecture to build

Six stages. Keep them as separate, individually-testable functions.

```
 call_NN.wav
     │
 ┌───▼──────────────────┐
 │ 1. Audio prep        │  resample 44.1k → 16k mono, check for silence/clipping
 └───┬──────────────────┘
 ┌───▼──────────────────┐
 │ 2. Transcription     │  German ASR + domain hint/vocabulary bias
 └───┬──────────────────┘  → save transcript to disk (cache!)
 ┌───▼──────────────────┐
 │ 3. Pre-normalization │  "ätt"→"@", "Bindestrich"→"-", "A wie Anton"→"A",
 └───┬──────────────────┘   number words → digits   (deterministic, unit-tested)
 ┌───▼──────────────────┐
 │ 4. LLM extraction    │  structured output → {first,last,email,phone} + confidence
 └───┬──────────────────┘
 ┌───▼──────────────────┐
 │ 5. Post-validation   │  email regex + domain check, phone → E.164-ish,
 └───┬──────────────────┘   name↔email cross-check, confidence score
 ┌───▼──────────────────┐
 │ 6. Output + eval     │  predictions.json  →  score vs ground_truth.json
 └──────────────────────┘
```

### Stage-by-stage

**1. Audio prep.**
The files are 44.1 kHz because they're synthetic; real telephony is 8 kHz. Most ASR APIs
want 16 kHz mono. One `ffmpeg` call. Also compute duration and RMS level — a 0-second or
silent file should fail *loudly and cheaply*, before you spend an API call on it.

**2. Transcription — the choice they want you to justify.**
Realistic options:

| Option | Notes |
|---|---|
| **Whisper `large-v3`** (hosted API or local `faster-whisper`) | Strong German. Accepts an `initial_prompt` / `prompt` — put your domain vocabulary in it. Local via `faster-whisper` = free and offline, ~real-time on an M-series Mac. |
| **Deepgram `nova-3`** (`language=de`) | Fast, cheap, and supports **keyterm boosting** — you can literally boost `gmail.com`, `gmx.de`, `Bindestrich`, `Unterstrich`. Very good fit for this task. |
| **AssemblyAI / ElevenLabs Scribe** | Also fine. Same idea. |
| **Gemini (audio-native)** | Can skip stage 2 entirely and go audio → JSON. Good as a *second opinion* path; see §6. |

The trick that separates a thoughtful submission from a basic one: **prime the ASR with
domain vocabulary.** Whisper's `prompt` parameter and Deepgram's `keyterm` both steer the
decoder. Feed it the 16 known domains, the spelling-alphabet words, and `Bindestrich /
Unterstrich / Klammeraffe / Punkt`. This alone typically moves email accuracy by double
digits.

Also ask for **word-level timestamps and per-word confidence** if the API offers them.
You get them for free and they become your monitoring signal later (§7).

**Always cache transcripts to disk.** `transcripts/call_07.json`. You'll iterate on the
prompt 20 times; don't re-transcribe every time. This costs you five lines and saves you
an hour.

**3. Pre-normalization — plain Python, no LLM.**
A dictionary + a few regexes converting spoken German into symbols and digits. This is
deterministic, free, instant, and **unit-testable**. Write ~10 tests here; it's the cheapest
"test coverage" box you can tick, and reviewers notice that you put tests where the risk is.

Don't destroy the original: pass **both** the raw transcript and the normalized one to the
LLM. The model is often better than your regex at resolving ambiguity, but it needs the
evidence.

**4. LLM extraction.**
One call per recording. Use **structured outputs** so the response is schema-valid JSON and
you never hand-parse a code fence. If you're using Claude, `claude-opus-5` is the default
pick ($5/$25 per Mtok) and `claude-sonnet-5` ($2/$10) is a sane cheaper comparison point —
which is itself a nice A/B experiment to show.

Ask for **more than the four fields**:

```jsonc
{
  "first_name": "Julia",
  "last_name": "Schröder",
  "email": "julia.schroeder@web.de",
  "phone_number": "+49 151 22998877",
  "confidence": { "first_name": 0.98, "email": 0.72, ... },
  "evidence": { "email": "j-u-l-i-a punkt s-c-h-r-o-e-d-e-r ätt web punkt de" },
  "notes": "caller spelled the email twice, second spelling used"
}
```

`evidence` (the exact transcript span each answer came from) is the highest-value extra
field. It makes every error debuggable in two seconds instead of two minutes, and it
visibly demonstrates "observability" without building any infrastructure.

Prompt must contain, explicitly:
- the German spelling conventions table from §3.1
- **the diacritics rule**: keep `ö/ü/í/'` in names, transliterate `ä→ae ö→oe ü→ue ß→ss` and
  strip accents in emails
- the phone output format (pick one and be consistent — see §5)
- `null` is a valid answer. *Never guess.* A wrong value is worse than a missing one,
  because a wrong email silently poisons a CRM record.
- 2–3 few-shot examples showing a messy transcript → correct JSON

**5. Post-validation.**
- email: RFC-ish regex; domain against the known-domain list → if it's a near-miss
  (`gmail.de` vs `gmail.com`, edit distance 1–2), **flag it**, and only auto-correct if you
  can defend the rule
- phone: strip to digits, normalize `0049`/`0` prefix → `+49`, then format consistently
- **name ↔ email cross-check**: `julia.schroeder@` vs `Julia Schröder` — after
  transliteration these should agree. When they disagree, one of the two is wrong and you
  know to flag the record. This is a genuinely clever check and it's nearly free.
- produce one `needs_human_review: true/false` per record

**6. Eval.**
Covered in §5 — and you should write this **second**, right after transcription, before you
tune anything.

---

## 5. The eval harness (write this early — it is the whole submission)

You cannot tune what you cannot measure. Build the scorer before you start improving
prompts, or you'll spend three hours on vibes.

`python -m phonebot.eval` should print something like:

```
field           exact   norm    n
first_name      29/30   30/30   96.7% / 100.0%
last_name       27/30   28/30   90.0% /  93.3%
email           22/30   24/30   73.3% /  80.0%
phone_number    25/30   30/30   83.3% / 100.0%
─────────────────────────────────────────────
overall        103/120  112/120 85.8% /  93.3%
full-record exact (4/4 correct): 19/30 (63.3%)

failures:
  call_12  email     got michael-wagner@gmx.de   want michael_wagner@gmx.de   (edit dist 1)
  call_02  email     got h47herbst@web.de        want h47-herbst@web.de       (edit dist 1)
```

Design notes:

**Normalize before comparing, and report both numbers.** Raw `==` on
`"+49 152 11223456"` vs `"+4915211223456"` fails, but those are the same phone number. So:

- phone: strip everything but digits and a leading `+`, then compare
- email: lowercase and strip whitespace
- names: strip/collapse whitespace; decide *explicitly* whether `Lisa-Marie` should match
  `Lisa Marie` (the brief's own example suggests yes) — and **write down the decision**

Report **exact** (strict string match) *and* **normalized** side by side. Showing the gap
between the two proves you thought about it, and protects you if their internal scorer is
stricter than yours.

**Log every failure with an edit distance.** An edit distance of 1 means "one character
away → fixable with a prompt tweak." An edit distance of 9 means "the ASR never heard it →
fix transcription instead." That number tells you which stage to spend your next 30 minutes
on. This is the single most useful diagnostic in the whole project.

**Report full-record accuracy too** (all 4 fields right). It's the metric the business
actually cares about — a CRM record with one wrong field is still a bad record.

**Don't overfit.** With 30 samples, "hardcode gmx.de" or "if transcript contains X then Y"
is visible and will be marked against you. Every rule you add should be one you'd defend
for call_31.

---

## 6. Error handling — where the "edge cases & failures" points live

Put these in the code, and name them in the README:

| Failure | Handling |
|---|---|
| API timeout / 5xx / 429 | Retry with exponential backoff + jitter. Cap attempts. The SDKs do 2 retries by default — set it deliberately, don't inherit it by accident. |
| Model returns invalid JSON | Structured outputs make this near-impossible; still, catch it and retry once with the parse error fed back. |
| Empty / silent / corrupt audio | Detect **before** the API call (duration ≈ 0, RMS below threshold). Return a record with all-null fields + `error: "silent_audio"`. Never crash the batch. |
| Caller never gave an email | `null`, `confidence: 0`. **Not** a hallucinated plausible address. |
| Caller corrects themselves ("nein, mit Doppel-s") | Prompt rule: *the last-stated value wins*; note the correction in `notes`. |
| One recording blows up | The other 29 still finish. Per-record try/except, error captured in the output row. Exit code non-zero if any failed. |
| Low confidence / failed cross-check | `needs_human_review: true` and route out of the automated path. |

Two things a reviewer will specifically look for:

1. **Partial failure doesn't kill the run.** Process all 30, report which ones broke.
2. **Every record is deterministic and re-runnable.** Set `temperature`/effort explicitly,
   cache transcripts, make `python -m phonebot.run` idempotent.

---

## 7. The discussion questions — prepare real answers

They will ask these live. Short, concrete answers beat long abstract ones.

**"How would you monitor this in production?"**
You have no ground truth in production — that's the whole problem. So you monitor
*proxies*:
- **Confidence distribution drift.** Mean email confidence dropping week-over-week = the
  bot script changed, or a new telco codec, or an ASR model update.
- **Validation-failure rate** — % of emails failing regex, % of phones failing length,
  % of name↔email cross-check disagreements. These need no labels.
- **Null rate per field** — a sudden spike in null emails means something upstream broke.
- **Human-review queue rate** — your actual cost driver, and the thing you optimize down.
- **Ground truth via feedback loop**: bounced emails and failed call-backs are *free labels*.
  A bounced email is a confirmed extraction failure. Wire that back.
- **A standing golden set** (these 30 + everything a human corrected) re-run on every prompt
  or model change, in CI. Never ship a prompt change without running it.
- Per-stage latency and cost per call; log `usage` from every API response.

**"How would you add a new entity — address, company name?"**
This is a question about *where the schema lives*. Right answer: entities are declared once,
in config, not scattered across the prompt and the parser.

```yaml
# entities.yaml
address:
  description: "Caller's postal address"
  type: string
  validator: german_address
  required: false
```

…and the prompt section, the JSON schema, the validator registry, and the eval columns are
all generated from that. Adding an entity = one YAML block + one validator function + labels
in the eval set. **No changes to pipeline code.** Say that out loud; it's exactly what
"Prompt Management: easy updates without code changes" in their brief is fishing for.

**"What metrics track system health?"**
Split them into three tiers and say so:
- *Quality*: per-field accuracy on the golden set, full-record accuracy, human-correction rate
- *Operational*: p50/p95 latency per stage, error rate by type, retry rate, throughput, queue depth
- *Business*: cost per call (ASR + LLM, tracked separately), % fully automated (no human
  touch), downstream email bounce rate, time-to-CRM

**"Prompt management / A/B testing"** (also in their brief, easy to satisfy cheaply):
- prompts live in versioned files (`prompts/extract_v1.md`, `v2.md`), never in string
  literals inside functions
- the run config names a prompt version + model; the output records which pair produced it
- your eval takes `--variant` and prints a comparison table → that *is* A/B testing at this
  scale. Run it once for real (e.g. with vs. without the normalization stage, or opus vs.
  sonnet) and put the resulting table in the README. A real measured comparison is worth
  more than a paragraph promising you could do one.

---

## 8. Suggested repo layout

```
phonebot/
├── README.md                  ← how to run + design decisions + YOUR RESULTS TABLE
├── pyproject.toml             ← uv/pip installable; pin versions
├── .env.example               ← names the env vars, contains no secrets
├── config.yaml                ← model ids, prompt version, paths, thresholds
├── prompts/
│   ├── extract_v1.md
│   └── extract_v2.md
├── src/phonebot/
│   ├── audio.py               ← prep + silence/corruption checks
│   ├── transcribe.py          ← ASR client + disk cache
│   ├── normalize.py           ← German spoken→symbol rules   ← unit-test this hard
│   ├── extract.py             ← LLM call, structured output, retries
│   ├── validate.py            ← email/phone/cross-check, confidence
│   ├── pipeline.py            ← orchestration, per-record error isolation
│   └── eval.py                ← scoring + failure report
├── tests/
│   ├── test_normalize.py      ← the highest-value tests in the project
│   ├── test_validate.py
│   └── test_eval.py           ← incl. the list-of-acceptable-values case
└── out/
    ├── transcripts/*.json
    ├── predictions.json
    └── scores.md
```

Two commands, documented at the top of the README:

```bash
uv sync && cp .env.example .env   # add your keys
python -m phonebot.run            # transcribe + extract all 30 → out/predictions.json
python -m phonebot.eval           # score against ground truth → out/scores.md
```

If a reviewer can't get to a score in under two minutes, you lose points you already earned.

---

## 9. Build order (this is your checklist)

Roughly 4–6 hours if you're new to it. Do it **in this order** — each step is useful even
if you run out of time at the next one.

- [ ] **0 · 15 min** — Listen to 3 recordings. Actually listen. `afplay call_02.wav`. You
      need to hear how the email gets spelled out; everything in §3 will click.
- [ ] **1 · 30 min** — Repo skeleton, config, `.env.example`, load `ground_truth.json`.
- [ ] **2 · 45 min** — Transcribe all 30, cache to `out/transcripts/`. Read a few by eye.
      You'll immediately see what your normalizer has to handle.
- [ ] **3 · 45 min** — **Write the eval harness now, against a dummy extractor.** Before
      any prompt tuning. This is the step people skip and it's the one that makes the rest
      efficient.
- [ ] **4 · 45 min** — v1 prompt + structured output. Run. Get your first real score.
      Expect ~85% names, ~50–65% emails. **Write the number down.**
- [ ] **5 · 60 min** — The normalization stage + prompt v2 with the spelling table, the
      diacritics rule, and few-shots drawn from your actual failures. Re-score.
      This is where most of your gain comes from.
- [ ] **6 · 30 min** — Validation, confidence, cross-check, `needs_human_review`. Re-score.
- [ ] **7 · 30 min** — Error handling: retries, silent-audio guard, per-record isolation.
      Unit tests for `normalize.py`.
- [ ] **8 · 30 min** — README: how to run, design decisions, **the results table**, the
      before/after from step 4 → 5, and "what I'd do next."
- [ ] **9 · 20 min** — Loom.

**If you're short on time, cut in this order:** the A/B variant flag, the confidence
scoring, the cross-check. **Never cut:** the eval harness, the README results table, or
error handling.

---

## 10. The Loom (they asked for it specifically — don't skip it)

5–8 minutes, screen share, don't rehearse it to death:

1. **(1 min)** The problem and your architecture — draw the six boxes from §4.
2. **(1 min)** Run it live. `python -m phonebot.eval`. Let them see the real numbers.
3. **(2 min)** The interesting part: German spelled-out emails, the diacritics
   name-vs-email split, the German spelling alphabet. Show one transcript and how it becomes
   a correct email. **This is the moment that says "senior."**
4. **(1.5 min)** Tradeoffs you chose and why: this ASR because X; normalize in code rather
   than asking the LLM because it's deterministic and testable; validate-don't-correct
   because `call_02`'s phone number would have been "fixed" into a wrong answer.
5. **(1.5 min)** Next iteration: ASR n-best candidates re-ranked by the LLM; a second
   audio-native model as a cross-check with disagreement → human review; active learning
   from human corrections; a real observability stack.

Say your accuracy number out loud, including where it's weak. Owning "emails are at 84% and
here's exactly which two fail and why" reads as far stronger than claiming 100%.

---

## 11. Ideas that separate a good submission from an average one

Cheap, high-signal, roughly in order of value-per-minute:

1. **Evidence spans in the output** (§4.4). Near-free, and it's the "observability" answer
   made concrete.
2. **Name ↔ email cross-check** (§4.5). A genuine consistency signal nobody else will have.
3. **A measured before/after table in the README** — "adding the normalization stage moved
   email accuracy 58% → 84%". Evidence that you iterate with data.
4. **`validate, don't correct`, justified with `call_02`'s over-long phone number.** Shows
   you actually read the data instead of reaching for a library.
5. **Two-model disagreement check** — run extraction twice (different model or different
   prompt); where they disagree, flag for review. Cheap ensemble, and it's a real production
   pattern.
6. **Handle the array-valued ground truth case** even though this dataset has none.
7. **ASR n-best / alternatives.** If the API returns alternatives, hand them all to the LLM
   and let it pick the one that forms a valid email. This is the advanced move and it maps
   directly onto how you'd actually push past 90%.

---

## 12. Mistakes to avoid

- ❌ Tuning prompts before the eval harness exists. You'll have no idea what helped.
- ❌ Hardcoding anything that only works on these 30 files. Reviewers grep for this.
- ❌ Strict phone validation that "fixes" `+49 157 231412313`. You'd lose a correct answer.
- ❌ Letting the model normalize `Schröder` → `Schroeder` in the **name** field. Read §3.2 again.
- ❌ A single `try/except Exception: pass`. That's a silent failure, and one of their five
  criteria is literally about failure handling.
- ❌ Re-transcribing on every run. Cache. You'll iterate 20+ times.
- ❌ Committing your API keys. `.env.example` with empty values; `.env` in `.gitignore`.
- ❌ A README that explains what the code does instead of **what you decided and why**.
  They can read the code. They can't read your reasoning.
- ❌ Shipping without the accuracy number. It's the first thing they'll look for.

---

## TL;DR

Build six small stages, cache the transcripts, and **write the scorer before the prompt**.
The challenge is won or lost on **German spelled-out emails** — the `@`/`Punkt`/
`Bindestrich`/`Unterstrich` mapping, the spelling alphabet, and the rule that names keep
their umlauts while emails transliterate them. Validate but never silently correct.
Put a real accuracy table in the README, and be honest about the weak field in the Loom.
