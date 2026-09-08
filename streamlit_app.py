import os
import re
import json
import time
import asyncio
import subprocess
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field

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

# Stable Flash Models
MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
]

# Strict Structured Output Schema
class DialogueSegment(BaseModel):
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    burmese: str = Field(description="Burmese translation of spoken text")

class DubbingResponse(BaseModel):
    segments: list[DialogueSegment]


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
        raise RuntimeError("Video/Audio duration ကို ဖတ်မရပါ။")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def extract_audio(video, out):
    r = run_cmd([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(out)
    ], 900)
    if r.returncode or not out.exists() or out.stat().st_size < 1000:
        raise RuntimeError("Original Audio ထုတ်ယူ၍ မရပါ။")


@st.cache_resource
def get_client():
    key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not key:
        raise RuntimeError("GEMINI_API_KEY မတွေ့ပါ။ Streamlit Secrets ထဲတွင် ထည့်ပေးပါ။")
    return genai.Client(api_key=key)


def analyze_audio_full(client, audio_path, status_box):
    status_box.update(label="AI ဘာသာပြန်ဆိုနေသည်...", state="running")
    uploaded = client.files.upload(file=str(audio_path))
    
    # Audio Upload Process အချိန်ပေးခြင်း
    time.sleep(3)

    total_dur = duration(audio_path)

    prompt = f"""
Listen to the audio file (total duration: {total_dur:.2f} seconds).
Transcribe all spoken dialogues and translate them into natural spoken Burmese for movie dubbing.
Provide accurate start and end timestamps in seconds.
"""

    for model in MODELS:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[uploaded, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=DubbingResponse,
                )
            )
            
            if resp.parsed and resp.parsed.segments:
                valid_segments = []
                for item in resp.parsed.segments:
                    if item.burmese and item.burmese.strip():
                        s = max(0.0, min(float(item.start), total_dur))
                        e = max(s + 0.1, min(float(item.end), total_dur))
                        valid_segments.append({
                            "start": s,
                            "end": e,
                            "burmese": item.burmese.strip()
                        })
                
                if valid_segments:
                    valid_segments.sort(key=lambda x: x["start"])
                    return valid_segments, model, total_dur
        except Exception:
            continue

    raise RuntimeError("Gemini API မှ စကားပြော ဖတ်ယူ၍ မရပါ။ API Key သို့မဟုတ် Video ဖိုင်၏ အသံကို စစ်ဆေးပါ။")


async def tts_async(text, voice, out):
    comm = edge_tts.Communicate(text=str(text).strip(), voice=voice)
    await comm.save(str(out))


def generate_tts(text, voice, out):
    try:
        if out.exists():
            out.unlink()
        asyncio.run(tts_async(text, voice, out))
        if not out.exists() or out.stat().st_size < 500:
            raise RuntimeError("TTS အသံ မထွက်ပါ။")
    except Exception as e:
        raise RuntimeError(f"TTS Error: {e}")


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


def fit_tts_audio(src, out, slot_duration):
    d = duration(src)
    speed = max(0.5, min(d / max(slot_duration, 0.25), 3.0))
    run_cmd([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-filter:a", atempo_filter(speed),
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(out)
    ], 180)


def build_final_audio(segments, voice, total_dur, work_dir, status_box):
    tts_files = []
    total = len(segments)

    for i, s in enumerate(segments, 1):
        status_box.update(label=f"မြန်မာအသံ ထုတ်လုပ်နေသည် ({i}/{total})...", state="running")
        raw = work_dir / f"tts_{i:04d}.mp3"
        fitted = work_dir / f"fit_{i:04d}.wav"

        generate_tts(s["burmese"], voice, raw)
        fit_tts_audio(raw, fitted, max(0.2, s["end"] - s["start"]))
        tts_files.append((s["start"], fitted))

    status_box.update(label="မြန်မာ Audio များကို ပေါင်းစပ်နေသည်...", state="running")

    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
    for _, f in tts_files:
        cmd += ["-i", str(f)]

    filters = []
    labels = []

    for i, (start_time, _) in enumerate(tts_files):
        ms = int(round(start_time * 1000))
        label = f"a{i}"
        filters.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,adelay={ms}|{ms}[{label}]")
        labels.append(f"[{label}]")

    filters.append("".join(labels) + f"amix=inputs={len(labels)}:duration=first:dropout_transition=0:normalize=0[mix]")
    out_audio = work_dir / "burmese_mixed.m4a"

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[mix]", "-t", f"{total_dur:.3f}",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(out_audio)
    ]

    r = run_cmd(cmd, 1800)
    if r.returncode or not out_audio.exists():
        raise RuntimeError(f"Audio Mixing မအောင်မြင်ပါ။\n{r.stderr}")

    return out_audio


def merge_video_audio(video_path, audio_path, output_path):
    r = run_cmd([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(output_path)
    ], 1800)

    if r.returncode:
        run_cmd([
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path), "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", str(output_path)
        ], 3600)


st.title("🎬 Movie Dubbing AI")
st.caption("Video တင်ပါ → Gemini AI dialogue ဖတ်ရှုမည် → မြန်မာ အသံထပ်ပေးမည်")

if "final_bytes" not in st.session_state:
    st.session_state.final_bytes = None

uploaded = st.file_uploader("🎥 Video တင်ပါ", type=["mp4", "mov", "mkv", "webm"])
voice_name = st.selectbox("🎙️ အသံ ရွေးချယ်ပါ", list(VOICES))
start = st.button("🚀 Start Dubbing", type="primary", use_container_width=True)

if start:
    st.session_state.final_bytes = None
    if not uploaded:
        st.error("Video တင်ပေးရန် လိုအပ်ပါသည်။")
        st.stop()

    with st.status("လုပ်ဆောင်နေပါသည်...", expanded=True) as status_box:
        try:
            with tempfile.TemporaryDirectory() as td:
                work_dir = Path(td)
                input_video = work_dir / "input_video.mp4"
                input_video.write_bytes(uploaded.getbuffer())

                status_box.update(label="1/4 Video မှ Audio ထုတ်ယူနေသည်...", state="running")
                orig_audio = work_dir / "orig_audio.wav"
                extract_audio(input_video, orig_audio)

                segments, used_model, total_dur = analyze_audio_full(get_client(), orig_audio, status_box)
                st.write(f"✅ AI Model: **{used_model}** | Dialogue စာကြောင်းရေ: **{len(segments)} လိုင်း**")

                burmese_audio = build_final_audio(segments, VOICES[voice_name], total_dur, work_dir, status_box)

                status_box.update(label="4/4 Video နှင့် Audio ပေါင်းစပ်နေသည်...", state="running")
                final_video = work_dir / "final_output.mp4"
                merge_video_audio(input_video, burmese_audio, final_video)

                st.session_state.final_bytes = final_video.read_bytes()
                status_box.update(label="Dubbing လုပ်ဆောင်မှု အောင်မြင်ပါသည်!", state="complete")

        except Exception as e:
            status_box.update(label="အမှားအယွင်း ဖြစ်ပေါ်ခဲ့ပါသည်။", state="error")
            st.exception(e)

if st.session_state.final_bytes:
    st.subheader("🎬 Preview")
    st.video(st.session_state.final_bytes)
    st.download_button(
        "⬇️ Download Dubbed Video",
        st.session_state.final_bytes,
        "dubbed_video.mp4",
        "video/mp4",
        use_container_width=True
    )
