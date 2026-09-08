import os
import re
import json
import time
import asyncio
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
import whisper
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
def get_gemini_client():
    key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not key:
        raise RuntimeError("GEMINI_API_KEY မတွေ့ပါ။ Streamlit Secrets ထဲတွင် ထည့်ပေးပါ။")
    return genai.Client(api_key=key)


@st.cache_resource
def load_whisper_model():
    # Fast & Light model for STT
    return whisper.load_model("base")


def transcribe_with_whisper(audio_path, status_box):
    status_box.update(label="1/4 OpenAI Whisper ဖြင့် Audio စကားပြော ဖတ်ယူနေသည်...", state="running")
    model = load_whisper_model()
    result = model.transcribe(str(audio_path))
    
    segments = []
    for s in result.get("segments", []):
        text = s.get("text", "").strip()
        if text:
            segments.append({
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": text
            })
    return segments


def translate_with_gemini(segments, client, status_box):
    status_box.update(label="2/4 Gemini AI ဖြင့် မြန်မာဘာသာသို့ ပြန်ဆိုနေသည်...", state="running")
    
    if not segments:
        raise RuntimeError("Whisper မှ ဗီဒီယိုထဲတွင် စကားပြော မတွေ့ရှိပါ။")

    input_json = json.dumps(segments, ensure_ascii=False)
    
    prompt = f"""
Translate the following speech dialogue segments into natural, movie-style spoken Burmese.
Keep the exact same JSON structure with keys: "start", "end", and "burmese" (translated text).

Input:
{input_json}

Return STRICTLY a JSON array of objects like this:
[
  {{"start": 0.5, "end": 2.1, "burmese": "မြန်မာစာသား"}}
]
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json"
        )
    )

    raw_text = getattr(resp, "text", "").strip()
    
    # Cleanup markdown JSON tags if present
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.I)
    raw_text = re.sub(r"\s*```$", "", raw_text)
    
    translated_data = json.loads(raw_text)
    return translated_data


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
        status_box.update(label=f"3/4 မြန်မာအသံ ထုတ်လုပ်နေသည် ({i}/{total})...", state="running")
        raw = work_dir / f"tts_{i:04d}.mp3"
        fitted = work_dir / f"fit_{i:04d}.wav"

        text = s.get("burmese") or s.get("text", "")
        generate_tts(text, voice, raw)
        fit_tts_audio(raw, fitted, max(0.2, float(s["end"]) - float(s["start"])))
        tts_files.append((float(s["start"]), fitted))

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


st.title("🎬 Movie Dubbing AI (Whisper + Gemini)")
st.caption("Whisper ဖြင့် Audio ကို စိတ်ချစွာဖတ်ရှုမည် → Gemini ဖြင့် မြန်မာပြန်မည် → အသံထပ်ပေးမည်")

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

                orig_audio = work_dir / "orig_audio.wav"
                extract_audio(input_video, orig_audio)
                total_dur = duration(orig_audio)

                # 1. Whisper Transcription
                whisper_segs = transcribe_with_whisper(orig_audio, status_box)
                
                # 2. Gemini Translation
                translated_segs = translate_with_gemini(whisper_segs, get_gemini_client(), status_box)
                st.write(f"✅ Dialogue စာကြောင်းရေ: **{len(translated_segs)} လိုင်း** တိကျစွာ ဖတ်ပြီးပါပြီ။")

                # 3. Audio Dubbing (100% Burmese Pure Stream)
                burmese_audio = build_final_audio(translated_segs, VOICES[voice_name], total_dur, work_dir, status_box)

                # 4. Merge Video and New Audio
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
