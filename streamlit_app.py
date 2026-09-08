import os
import re
import json
import time
import asyncio
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
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]


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


def clean_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("["), text.rfind("]")
    if a >= 0 and b > a:
        return text[a:b + 1]
    return text


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


def analyze_chunk(client, chunk_file, offset, chunk_dur):
    uploaded = client.files.upload(file=str(chunk_file))
    time.sleep(1)

    prompt = f"""
Transcribe EVERY spoken sentence in this audio snippet (duration: {chunk_dur:.2f}s).
Do NOT summarize. Translate each sentence into natural spoken Burmese for video dubbing.

Return strictly a JSON array of objects with the key "burmese":
[
  {{"burmese": "မြန်မာစာသား"}}
]
"""
    for model in MODELS:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_uri(file_uri=uploaded.uri, mime_type="audio/wav"), prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            raw_text = getattr(resp, "text", "")
            if not raw_text:
                continue
            raw_json = json.loads(clean_json(raw_text))
            
            lines = []
            if isinstance(raw_json, list):
                for item in raw_json:
                    if isinstance(item, dict) and item.get("burmese"):
                        lines.append(str(item["burmese"]).strip())

            if lines:
                # တွေ့ရှိသမျှ စကားပြောများကို Chunk အတွင်း အလိုက်သင့် အချိန်ခွဲဝေပေးခြင်း
                slot_time = chunk_dur / len(lines)
                segs = []
                for i, text in enumerate(lines):
                    segs.append({
                        "start": offset + (i * slot_time),
                        "end": offset + ((i + 1) * slot_time),
                        "burmese": text
                    })
                return segs, model
        except Exception:
            continue
    return [], MODELS[0]


def analyze_audio_full(client, audio_path, status_box):
    chunks, total_dur = split_audio_chunks(audio_path, chunk_length=15.0)
    all_segments = []
    used_model = MODELS[0]

    for idx, (offset, chunk_dur, chunk_file) in enumerate(chunks, start=1):
        status_box.update(label=f"AI dialogue ဖတ်နေသည် Chunk ({idx}/{len(chunks)})...", state="running")
        segs, m = analyze_chunk(client, chunk_file, offset, chunk_dur)
        used_model = m
        all_segments.extend(segs)

    if not all_segments:
        raise RuntimeError("Video ထဲမှ စကားပြော Dialogue များ ဖတ်ယူ၍ မရပါ။")

    return all_segments, used_model, total_dur


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


def build_final_audio(segments, voice, total_dur, orig_audio, work_dir, status_box):
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
st.caption("Video တင်ပါ → Absolute Chunk AI dialogue ဖတ်ရှုမည် → မြန်မာ အသံထပ်ပေးမည်")

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
                st.write(f"✅ AI Model: **{used_model}** | Dialogue စာကြောင်းရေ: **{len(segments)} လိုင်း** (အစမှ အဆုံး အပြည့်အဝ ဖတ်ပြီး)")

                burmese_audio = build_final_audio(segments, VOICES[voice_name], total_dur, orig_audio, work_dir, status_box)

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
