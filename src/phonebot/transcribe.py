"""OpenAI file transcription with a content- and settings-aware cache.

The cache key is sha256(audio bytes + request settings), so re-running is free until
either the audio or a transcription setting actually changes. Audio is probed locally
first: a file that is too short or effectively silent is rejected before it costs an
API call.
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any

ASR_PROMPT = (
    "Telefonat auf Deutsch. Kontaktdaten, E-Mail-Adressen und Telefonnummern. "
    "Buchstabieren: at, Klammeraffe, Punkt, Bindestrich, Unterstrich, "
    "Doppel-s, zwo. Namen können international sein."
)

MIN_DURATION_S = 1.0
# Full scale is 32767 for 16-bit PCM; a real call sits far above this floor.
MIN_RMS = 50.0

_SAMPLE_TYPE = {1: "b", 2: "h", 4: "i"}


class AudioRejected(ValueError):
    """Local audio checks failed, so no API call was attempted."""


def probe(wav: Path) -> dict[str, float]:
    """Duration and RMS amplitude, read straight from the PCM frames."""
    with wave.open(str(wav)) as audio:
        frames, rate = audio.getnframes(), audio.getframerate()
        width, channels = audio.getsampwidth(), audio.getnchannels()
        raw = audio.readframes(frames)

    typecode = _SAMPLE_TYPE.get(width)
    if typecode is None:
        raise AudioRejected(f"unsupported_sample_width:{width}")
    samples = array.array(typecode)
    samples.frombytes(raw[: len(raw) - len(raw) % samples.itemsize])
    if width == 1:  # 8-bit PCM is unsigned; centre it on zero
        samples = array.array("h", (s - 128 for s in samples))

    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0
    return {
        "duration": frames / rate if rate else 0.0,
        "rms": rms,
        "sample_rate": float(rate),
        "channels": float(channels),
    }


def _cache_path(wav: Path, cache_dir: Path, audio_hash: str, settings: dict) -> Path:
    key = hashlib.sha256(
        (audio_hash + json.dumps(settings, sort_keys=True)).encode()
    ).hexdigest()[:16]
    return cache_dir / f"{wav.stem}-{key}.json"


def transcribe(
    wav: Path,
    cache_dir: Path,
    *,
    model: str,
    language: str,
    force: bool = False,
    client: Any | None = None,
) -> dict:
    """Transcribe one recording, reusing the cached result when nothing changed.

    `client` is injectable so tests can exercise this without network or credentials.
    """
    stats = probe(wav)
    if stats["duration"] < MIN_DURATION_S:
        raise AudioRejected(f"audio_too_short:{stats['duration']:.2f}s")
    if stats["rms"] < MIN_RMS:
        raise AudioRejected(f"audio_silent:rms={stats['rms']:.1f}")

    settings = {
        "model": model,
        "language": language,
        "prompt": ASR_PROMPT,
        "response_format": "json",
    }
    audio_hash = hashlib.sha256(wav.read_bytes()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _cache_path(wav, cache_dir, audio_hash, settings)
    if cached.exists() and not force:
        result = json.loads(cached.read_text())
        result["cached"] = True
        return result

    owned = client is None
    if owned:
        from openai import OpenAI  # imported lazily: tests inject a client instead

        client = OpenAI(timeout=60.0, max_retries=2)
    try:
        with wav.open("rb") as handle:
            response = client.audio.transcriptions.create(file=handle, **settings)
    finally:
        if owned:
            client.close()

    text = (response.text or "").strip()
    if not text:
        raise ValueError("empty_transcript")

    result = {
        "id": wav.stem,
        "text": text,
        "duration": round(stats["duration"], 2),
        "rms": round(stats["rms"], 1),
        "audio_hash": audio_hash,
        "settings": settings,
    }
    cached.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    result["cached"] = False
    return result


def main() -> int:
    """Transcribe every recording named in config.yaml; print a one-line summary each."""
    import sys

    import yaml
    from dotenv import load_dotenv

    load_dotenv()
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    paths, asr = cfg["paths"], cfg["asr"]
    cache_dir = Path(paths["transcripts"])

    failed = 0
    for wav in sorted(Path(paths["recordings"]).glob("*.wav")):
        try:
            result = transcribe(wav, cache_dir, model=asr["model"], language=asr["language"])
        except Exception as exc:  # one bad recording must not kill the batch
            failed += 1
            print(f"! {wav.stem}  {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        mark = "=" if result.get("cached") else "+"
        print(f"{mark} {result['id']}  {result['text'][:110]}")
    if failed:
        print(f"\n{failed} recording(s) failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
