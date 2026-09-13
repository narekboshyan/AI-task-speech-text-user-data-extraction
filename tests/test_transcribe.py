"""Offline tests for the transcription cache and the local audio guards."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from phonebot.transcribe import ASR_PROMPT, AudioRejected, probe, transcribe

RATE = 16000


def write_wav(path: Path, seconds: float, amplitude: int) -> Path:
    frames = int(RATE * seconds)
    samples = (
        int(amplitude * math.sin(2 * math.pi * 440 * i / RATE)) for i in range(frames)
    )
    with wave.open(str(path), "w") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return path


class FakeClient:
    """Stands in for the OpenAI client; counts calls so we can prove caching."""

    def __init__(self, text: str = "Guten Tag, hier ist Julia Schröder.") -> None:
        self.calls: list[dict] = []
        self._text = text
        self.audio = type("Audio", (), {"transcriptions": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"text": self._text})()

    def close(self) -> None:  # pragma: no cover - never used, client is injected
        raise AssertionError("injected client must not be closed by transcribe()")


def test_probe_reports_duration_and_rms(tmp_path):
    stats = probe(write_wav(tmp_path / "tone.wav", 2.0, 8000))
    assert stats["duration"] == pytest.approx(2.0, abs=0.01)
    assert stats["rms"] == pytest.approx(8000 / math.sqrt(2), rel=0.02)


def test_short_audio_is_rejected_before_any_api_call(tmp_path):
    client = FakeClient()
    with pytest.raises(AudioRejected, match="audio_too_short"):
        transcribe(write_wav(tmp_path / "blip.wav", 0.4, 8000), tmp_path / "cache",
                   model="m", language="de", client=client)
    assert client.calls == []


def test_silent_audio_is_rejected_before_any_api_call(tmp_path):
    client = FakeClient()
    with pytest.raises(AudioRejected, match="audio_silent"):
        transcribe(write_wav(tmp_path / "quiet.wav", 3.0, 5), tmp_path / "cache",
                   model="m", language="de", client=client)
    assert client.calls == []


def test_second_run_is_served_from_cache(tmp_path):
    wav = write_wav(tmp_path / "call.wav", 2.0, 8000)
    cache, client = tmp_path / "cache", FakeClient()

    first = transcribe(wav, cache, model="m", language="de", client=client)
    second = transcribe(wav, cache, model="m", language="de", client=client)

    assert len(client.calls) == 1
    assert client.calls[0]["prompt"] == ASR_PROMPT
    assert client.calls[0]["language"] == "de"
    assert first["cached"] is False and second["cached"] is True
    assert second["text"] == first["text"] == "Guten Tag, hier ist Julia Schröder."


def test_changed_settings_and_force_both_bypass_the_cache(tmp_path):
    wav = write_wav(tmp_path / "call.wav", 2.0, 8000)
    cache, client = tmp_path / "cache", FakeClient()

    transcribe(wav, cache, model="m", language="de", client=client)
    transcribe(wav, cache, model="other", language="de", client=client)
    transcribe(wav, cache, model="m", language="de", force=True, client=client)

    assert len(client.calls) == 3


def test_different_audio_does_not_reuse_a_cached_transcript(tmp_path):
    cache, client = tmp_path / "cache", FakeClient()
    transcribe(write_wav(tmp_path / "a.wav", 2.0, 8000), cache, model="m",
               language="de", client=client)
    transcribe(write_wav(tmp_path / "a.wav", 3.0, 8000), cache, model="m",
               language="de", client=client)
    assert len(client.calls) == 2


def test_empty_transcript_raises(tmp_path):
    with pytest.raises(ValueError, match="empty_transcript"):
        transcribe(write_wav(tmp_path / "call.wav", 2.0, 8000), tmp_path / "cache",
                   model="m", language="de", client=FakeClient(text="   "))
