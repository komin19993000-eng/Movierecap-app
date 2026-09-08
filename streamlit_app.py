import os
import re
import json
import time
import asyncio
import random
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
import edge_tts
import imageio_ffmpeg

st.set_page_config(
    page_title="Movie Dubbing AI",
    page_icon="🎬",
    layout="wide"
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

VOICES = {
    "သီဟ (အမျိုးသား)": "my-MM-ThihaNeural",
    "နီလာ (အမျိုးသမီး)": "my-MM-NilarNeural",
}

MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
]

TEMP_WORDS = (
    "503", "500", "502", "504", "429", "timeout",
    "unavailable", "overloaded", "high demand", "resource exhausted"
)


def run_cmd(args, timeout=1800):
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout
    )


def duration(path):
    r = run_cmd([FFMPEG, "-hide_banner", "-i", str(path)], 120)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr or "")
    if not m:
        raise RuntimeError("Video/audio duration ကို ဖတ်မရပါ။")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def extract_audio(video, out):
    r = run_cmd([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-vn", "-ac", "2", "-ar", "48000",
        "-c:a", "pcm_s16le", str(out)
    ], 900)
    if r.returncode or not out.exists() or out.stat().st_size < 1000:
        raise RuntimeError("Original audio ထုတ်မရပါ။\n" + (r.stderr or ""))


@st.cache_resource
def client():
    key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not key:
        raise RuntimeError("GEMINI_API_KEY မတွေ့ပါ။ Streamlit Secrets ထဲမှာ ထည့်ပါ။")
    return genai.Client(api_key=key)


def temporary_error(e):
    s = str(e).lower()
    return any(x.lower() in s for x in TEMP_WORDS)


def model_list(c):
    try:
        names = [getattr(m, "name", "").replace("models/", "") for m in c.models.list() if getattr(m, "name", "")]
        usable = [m for m in MODELS if m in names]
        return usable or MODELS
    except Exception:
        return MODELS


def clean_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("["), text.rfind("]")
    if a >= 0 and b > a:
        return text[a:b + 1]
    return text


def normalize(items, max_dur):
    out = []
    if not isinstance(items, list):
        return out

    for x in items:
        if not isinstance(x, dict):
            continue
        try:
            s, e = float(x.get("start", 0)), float(x.get("end", 0))
        except Exception:
            continue
        t = str(x.get("burmese", "")).strip()
        s, e = max(0, min(s, max_dur)), max(0, min(e, max_dur))

        if t and e - s >= 0.15:
            out.append({"start": s, "end": e, "burmese": t})

    out.sort(key=lambda x: x["start"])
    cleaned = []
    for x in out:
        if cleaned and abs(x["start"] - cleaned[-1]["start"]) < 0.1 and x["burmese"] == cleaned[-1]["burmese"]:
            cleaned[-1]["end"] = max(cleaned[-1]["end"], x["end"])
        else:
            cleaned.append(x)
    return cleaned


def split_audio_chunks(audio_path, chunk_length=15.0):
    total_dur = duration(audio_path)
    chunks = []
    start = 0.0
    idx = 0
    work_dir = audio_path.parent

    while start < total_dur:
        end = min(start + chunk_length, total_dur)
        chunk_file = work_dir / f"chunk_{idx:03d}.wav"
        run_cmd([
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(audio_path),
            "-t", f"{(end - start):.3f}", "-c:a", "pcm_s16le", str(chunk_file)
        ], 120)
        chunks.append((start, end - start, chunk_file))
        start += chunk_length
        idx += 1
    return chunks, total_dur


def analyze_chunk(c, chunk_file, offset, chunk_dur, status):
    uploaded = c.files.upload(file=str(chunk_file))
    prompt = f"""
Transcribe EVERY SINGLE word spoken in this audio snippet (duration {chunk_dur:.2f}s).
Do NOT summarize. Do NOT drop any phrase or word.
Provide start/end relative to this snippet in seconds.
Translate into natural Burmese for dubbing.

Return strictly JSON list:
[
  {{"start": 0.1, "end": 2.5, "burmese": "မြန်မာဘာသာပြန်"}}
]
"""
    for model in model_list(c):
        for attempt in range(2):
            try:
                resp = c.models.generate_content(
                    model=model,
                    contents=[types.Part.from_uri(file_uri=uploaded.uri, mime_type="audio/wav"), prompt],
                    config=types.GenerateContentConfig(temperature=0.10)
                )
                raw_json = json.loads(clean_json(getattr(resp, "text", "")))
                segments = normalize(raw_json, chunk_dur)
                for s in segments:
                    s["start"] += offset
                    s["end"] += offset
                return segments, model
            except Exception as e:
                if attempt == 0 and temporary_error(e):
                    time.sleep(2)
                else:
                    break
    return [], MODELS[0]


def analyze(c, audio, status):
    chunks, total_dur = split_audio_chunks(audio, chunk_length=15.0)
    all_segments = []
    used_model = "gemini-3.5-flash"

    for idx, (offset, chunk_dur, chunk_file) in enumerate(chunks, start=1):
        status(f"AI dialogue ဖတ်နေသည် Chunk {idx}/{len(chunks)}...")
        segs, m = analyze_chunk(c, chunk_file, offset, chunk_dur, status)
        used_model = m
        all_segments.extend(segs)

    all_segments = normalize(all_segments, total_dur)
    if not all_segments:
        raise RuntimeError("Video မှ စကားပြော Dialogue မတွေ့ရှိပါ သို့မဟုတ် AI response မရရှိပါ။")
    return all_segments, used_model


async def tts_async(text, voice, out):
    communicate = edge_tts.Communicate(text=str(text).strip(), voice=voice)
    await communicate.save(str(out))


def tts(text, voice, out):
    text = str(text).strip()
    if not text:
        raise RuntimeError("TTS text အလွတ်ဖြစ်နေပါသည်။")
    voices = [voice] + [v for v in VOICES.values() if v != voice]
    for selected_voice in voices:
        for attempt in range(3):
            try:
                if out.exists():
                    out.unlink()
                asyncio.run(tts_async(text, selected_voice, out))
                if out.exists() and out.stat().st_size >= 1000:
                    return selected_voice
            except Exception:
                if out.exists():
                    try: out.unlink()
                    except Exception: pass
                if attempt < 2:
                    time.sleep(1 + attempt)
    raise RuntimeError("Edge TTS အသံထုတ်ယူ၍ မရပါ။")


def atempo_filter(speed):
    speed = max(0.5, min(float(speed), 3.0))
    f = []
    while speed > 2:
        f.append("atempo=2")
        speed /= 2
    while speed < 0.5:
        f.append("atempo=.5")
        speed /= 0.5
    f.append(f"atempo={speed:.6f}")
    return ",".join(f)


def fit_tts(src, out, slot):
    d = duration(src)
    speed = max(0.5, min(d / max(slot, 0.25), 3.0))
    run_cmd([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-filter:a", atempo_filter(speed),
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(out)
    ], 180)


def validate_audio(path):
    r = run_cmd([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:a:0", "-f", "null", "-"], 300)
    return r.returncode == 0


def make_burmese_audio(segments, voice, dur, orig_audio, work, progress):
    files = []
    total = len(segments)

    for i, s in enumerate(segments, 1):
        progress((i - 1) / max(total, 1), f"TTS Voice ပြုလုပ်နေသည် {i}/{total}")
        raw = work / f"tts_{i:04d}.mp3"
        fitted = work / f"fit_{i:04d}.wav"

        tts(s["burmese"], voice, raw)
        fit_tts(raw, fitted, max(0.25, s["end"] - s["start"]))
        files.append((s["start"], fitted))

    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(orig_audio)]
    for _, f in files:
        cmd += ["-i", str(f)]

    filters = ["[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.25[bg]"]
    labels = ["[bg]"]

    for i, (start, _) in enumerate(files, start=1):
        ms = int(round(start * 1000))
        label = f"a{i}"
        filters.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,adelay={ms}|{ms}[{label}]")
        labels.append(f"[{label}]")

    filters.append("".join(labels) + f"amix=inputs={len(labels)}:duration=first:dropout_transition=0:normalize=0[mix]")
    out = work / "burmese_audio.m4a"

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[mix]", "-t", f"{dur:.3f}",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(out)
    ]

    r = run_cmd(cmd, 1800)
    if r.returncode or not out.exists() or not validate_audio(out):
        raise RuntimeError("Burmese audio mixing မအောင်မြင်ပါ။\n" + (r.stderr or ""))

    progress(1.0, "Burmese audio OK")
    return out


def export_video(video, audio, out):
    r = run_cmd([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-ar", "48000", "-ac", "2", "-shortest", "-movflags", "+faststart", str(out)
    ], 1800)

    if r.returncode:
        run_cmd([
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-ar", "48000", "-ac", "2", "-shortest", "-movflags", "+faststart", str(out)
        ], 3600)


st.title("🎬 Movie Dubbing AI")
st.caption("Video → AI dialogue (Chunk-based) → Burmese dubbing → Download")

if "final_bytes" not in st.session_state:
    st.session_state.final_bytes = None

uploaded = st.file_uploader("🎥 Video တင်ပါ", type=["mp4", "mov", "mkv", "webm"])
voice_name = st.selectbox("🎙️ အသံ", list(VOICES))
start = st.button("🚀 Start Dubbing", type="primary", use_container_width=True)

if start:
    st.session_state.final_bytes = None
    if not uploaded:
        st.error("Video တစ်ခုအရင်တင်ပါ။")
        st.stop()

    try:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            video = work / "input_video"
            video.write_bytes(uploaded.getbuffer())

            dur = duration(video)
            status = st.empty()
            bar = st.progress(0.0)

            status.info("1/5 Video စစ်ဆေးနေသည်...")
            audio = work / "original_audio.wav"
            extract_audio(video, audio)
            bar.progress(0.10)

            seg, model = analyze(client(), audio, status)
            bar.progress(0.35)
            st.success(f"AI model: {model} • Total Dialogue: {len(seg)} lines (အပြည့်အဝ ဖတ်ပြီး)")

            def prog(p, t):
                bar.progress(0.35 + 0.45 * p)
                status.info(t)

            burmese = make_burmese_audio(seg, VOICES[voice_name], dur, audio, work, prog)
            status.info("4/5 Final video ပြုလုပ်နေသည်...")
            final = work / "final_dubbed.mp4"

            export_video(video, burmese, final)
            bar.progress(1.0)

            st.session_state.final_bytes = final.read_bytes()
            status.success("5/5 Dubbing ပြီးပါပြီ!")

    except Exception as e:
        st.exception(e)

if st.session_state.final_bytes:
    st.subheader("🎬 Preview")
    st.video(st.session_state.final_bytes)
    st.download_button("⬇️ Download Final Video", st.session_state.final_bytes, "dubbed_video.mp4", "video/mp4", use_container_width=True)
