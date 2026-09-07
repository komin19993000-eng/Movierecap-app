# streamlit_app.py
# AI Movie Recap / Burmese Dubbing App
# Streamlit Cloud compatible

import os
import re
import json
import time
import asyncio
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
from google import genai
import edge_tts


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Movie Recap",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .sub-title {
        color: #888;
        margin-bottom: 25px;
    }

    .stage {
        padding: 12px 16px;
        border-radius: 12px;
        margin: 6px 0;
        border: 1px solid rgba(128,128,128,.2);
    }

    .success-box {
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(0,200,100,.3);
        background: rgba(0,200,100,.07);
    }

    .error-box {
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,0,0,.3);
        background: rgba(255,0,0,.07);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🎬 AI Movie Recap Automator</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">Natural Burmese dubbing • Smart dialogue timing • AI translation</div>',
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

VOICE_MAP = {
    "အမျိုးသား (သီဟ)": "my-MM-ThihaNeural",
    "အမျိုးသမီး (နီလာ)": "my-MM-NilarNeural",
}

# Current stable model first, older stable fallbacks after it.
MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "processing": False,
    "result_bytes": None,
    "result_name": None,
    "script": [],
    "logs": [],
    "selected_model": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def run_cmd(command, timeout=3600):
    """Run command safely and return stdout."""
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-5000:] if result.stderr else "FFmpeg command failed."
        )

    return result.stdout


def ffmpeg_exists():
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_duration(path):
    output = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout=60,
    )

    return float(output.strip())


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clean_json_text(text):
    """Extract JSON from Gemini response."""
    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    start = text.find("[")
    end = text.rfind("]")

    if start >= 0 and end > start:
        return text[start:end + 1]

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        return text[start:end + 1]

    return text


def parse_segments(text):
    raw = clean_json_text(text)

    data = json.loads(raw)

    if isinstance(data, dict):
        if "segments" in data:
            data = data["segments"]
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError("Gemini did not return a valid segment list.")

    result = []

    for item in data:
        if not isinstance(item, dict):
            continue

        start = safe_float(
            item.get("start", item.get("start_time", 0))
        )

        end = safe_float(
            item.get("end", item.get("end_time", 0))
        )

        text_value = str(
            item.get(
                "text",
                item.get(
                    "burmese",
                    item.get("translation", ""),
                ),
            )
        ).strip()

        if end <= start:
            continue

        if not text_value:
            continue

        result.append(
            {
                "start": max(0.0, start),
                "end": max(0.0, end),
                "text": text_value,
            }
        )

    result.sort(key=lambda x: x["start"])

    cleaned = []

    last_end = -1

    for item in result:
        if item["start"] < last_end:
            item["start"] = last_end

        if item["end"] <= item["start"]:
            continue

        cleaned.append(item)
        last_end = item["end"]

    if not cleaned:
        raise ValueError("No usable dialogue segments were returned.")

    return cleaned


def get_available_models(client):
    """Return models that are actually visible to this API key."""
    available = []

    try:
        for model in client.models.list():
            name = getattr(model, "name", "") or ""

            if name.startswith("models/"):
                name = name[7:]

            available.append(name)

    except Exception:
        return []

    return available


def choose_model(client):
    available = get_available_models(client)

    if available:
        for candidate in MODEL_CANDIDATES:
            if candidate in available:
                return candidate

    # If model listing is unavailable, try known stable models.
    return MODEL_CANDIDATES


def gemini_generate(client, contents, config=None):
    models = choose_model(client)

    if isinstance(models, str):
        models = [models]

    errors = []

    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            text = getattr(response, "text", None)

            if text:
                st.session_state.selected_model = model
                return text

        except Exception as exc:
            errors.append(f"{model}: {exc}")

    raise RuntimeError(
        "Gemini AI failed.\n\n" + "\n".join(errors[-5:])
    )


def extract_audio(video_path, audio_path):
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        timeout=3600,
    )


def upload_to_gemini(client, audio_path):
    uploaded = client.files.upload(
        file=str(audio_path)
    )

    return uploaded


def wait_for_file(client, uploaded):
    name = getattr(uploaded, "name", None)

    if not name:
        return uploaded

    for _ in range(60):
        try:
            current = client.files.get(name=name)

            state = getattr(current, "state", None)

            state_name = str(
                getattr(state, "name", state)
            ).upper()

            if "PROCESSING" not in state_name:
                return current

        except Exception:
            pass

        time.sleep(2)

    return uploaded


# ============================================================
# GEMINI TRANSCRIPTION + TRANSLATION
# ============================================================

def transcribe_and_translate(client, audio_file):
    prompt = r"""
You are a professional Burmese movie recap narrator.

Listen to the entire uploaded movie audio carefully.

Your task:

1. Detect every meaningful spoken dialogue.
2. Ignore music, sound effects, silence and background noise.
3. Identify the exact start and end time of each spoken dialogue.
4. Translate the dialogue into natural spoken Burmese.
5. Do NOT translate word-for-word.
6. Make the Burmese sound like a professional Myanmar movie recap narrator.
7. Preserve the original meaning, emotion, intention and context.
8. Keep each sentence short enough to be spoken naturally.
9. Do not invent dialogue.
10. Do not skip important dialogue.
11. Keep chronological order.
12. Use Burmese script.

Return ONLY valid JSON.

Format:

[
  {
    "start": 0.00,
    "end": 3.50,
    "text": "မြန်မာဘာသာဖြင့် သဘာဝကျကျ ပြန်ဆိုထားသော စကား"
  }
]

Important:
- start and end must be seconds.
- Do not use "s".
- Do not use markdown.
- Do not include explanations.
"""

    uploaded = wait_for_file(client, audio_file)

    response_text = gemini_generate(
        client,
        [
            uploaded,
            prompt,
        ],
    )

    return parse_segments(response_text)


# ============================================================
# TTS
# ============================================================

async def create_tts_async(text, voice, output_path):
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
        volume="+0%",
        pitch="+0Hz",
    )

    await communicator.save(str(output_path))


def create_tts(text, voice, output_path):
    asyncio.run(
        create_tts_async(
            text,
            voice,
            output_path,
        )
    )


# ============================================================
# AUDIO SPEED
# ============================================================

def atempo_filter(speed):
    """
    FFmpeg atempo supports 0.5 - 2.0.
    Build a chain when needed.
    """

    speed = max(0.25, min(4.0, speed))

    parts = []

    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5

    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0

    parts.append(f"atempo={speed:.6f}")

    return ",".join(parts)


def fit_audio_to_slot(input_audio, output_audio, target_duration):
    """
    Fit TTS into the dialogue slot without changing video duration.
    """

    source_duration = get_duration(input_audio)

    if source_duration <= 0:
        raise RuntimeError("Invalid TTS duration.")

    target_duration = max(0.05, target_duration)

    speed = source_duration / target_duration

    # Don't create extreme speech.
    # If it would be too fast, cap the speed and let the voice
    # slightly overlap only within the dialogue boundary.
    speed = max(0.55, min(2.0, speed))

    filter_value = atempo_filter(speed)

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_audio),
            "-filter:a",
            filter_value,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_audio),
        ],
        timeout=300,
    )


# ============================================================
# CREATE FULL DUBBED AUDIO
# ============================================================

def build_dubbed_audio(
    segments,
    voice,
    total_duration,
    work_dir,
    progress_callback=None,
):
    """
    Create one continuous Burmese audio track.

    Every dialogue is positioned at its original timestamp.
    The video itself is NOT sped up or slowed down.
    """

    audio_files = []

    total = len(segments)

    for index, segment in enumerate(segments):
        text = segment["text"]
        start = segment["start"]
        end = segment["end"]

        slot = end - start

        if slot <= 0:
            continue

        raw_tts = work_dir / f"tts_{index:05d}.mp3"
        fitted = work_dir / f"fit_{index:05d}.m4a"

        create_tts(
            text,
            voice,
            raw_tts,
        )

        fit_audio_to_slot(
            raw_tts,
            fitted,
            slot,
        )

        audio_files.append(
            {
                "path": fitted,
                "start": start,
                "end": end,
            }
        )

        if progress_callback:
            progress_callback(
                0.55 + (0.25 * ((index + 1) / max(1, total))),
                f"အသံဖန်တီးနေသည်... {index + 1}/{total}",
            )

    if not audio_files:
        raise RuntimeError("No Burmese voice was generated.")

    # --------------------------------------------------------
    # Create silence base
    # --------------------------------------------------------

    silence_base = work_dir / "silence.m4a"

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            f"{total_duration:.3f}",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(silence_base),
        ],
        timeout=300,
    )

    # --------------------------------------------------------
    # Mix every dialogue at absolute timestamp
    # --------------------------------------------------------

    inputs = [
        "-i",
        str(silence_base),
    ]

    for item in audio_files:
        inputs.extend(
            [
                "-i",
                str(item["path"]),
            ]
        )

    filters = []

    # Base silence
    filters.append(
        "[0:a]aresample=48000,asetpts=PTS-STARTPTS[base]"
    )

    mix_inputs = ["[base]"]

    for index, item in enumerate(audio_files, start=1):
        delay_ms = int(round(item["start"] * 1000))

        filters.append(
            f"[{index}:a]"
            f"aresample=48000,"
            f"asetpts=PTS-STARTPTS,"
            f"adelay={delay_ms}|{delay_ms}"
            f"[a{index}]"
        )

        mix_inputs.append(f"[a{index}]")

    filters.append(
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:"
          f"duration=first:"
          f"dropout_transition=0:"
          f"normalize=0,"
          f"atrim=0:{total_duration:.3f},"
          f"asetpts=PTS-STARTPTS,"
          f"loudnorm=I=-16:TP=-1.5:LRA=11,"
          f"aformat=sample_rates=48000:channel_layouts=stereo"
          f"[out]"
    )

    output_audio = work_dir / "burmese_full.m4a"

    command = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_audio),
    ]

    run_cmd(
        command,
        timeout=3600,
    )

    return output_audio


# ============================================================
# FINAL VIDEO
# ============================================================

def merge_video_audio(video_path, audio_path, output_path):
    """
    Keep original video exactly as it is.
    Remove original audio.
    Add Burmese audio.
    """

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        timeout=3600,
    )


# ============================================================
# PROGRESS UI
# ============================================================

def make_progress():
    progress = st.progress(0)
    status = st.empty()
    eta = st.empty()

    started = time.time()

    def update(value, message):
        value = max(0.0, min(1.0, value))

        progress.progress(
            int(value * 100)
        )

        elapsed = time.time() - started

        if value > 0.01:
            remaining = elapsed * (1.0 - value) / value

            if remaining < 60:
                eta_text = f"{remaining:.0f} sec"
            else:
                eta_text = f"{remaining / 60:.1f} min"

            eta.markdown(
                f"**Progress:** {value * 100:.0f}%  •  "
                f"**ETA:** {eta_text}"
            )
        else:
            eta.markdown("**Progress:** 0%")

        status.markdown(
            f"**{message}**"
        )

    return update


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    voice_name = st.selectbox(
        "🎙️ Burmese Voice",
        list(VOICE_MAP.keys()),
    )

    st.markdown("---")

    st.markdown("### Pipeline")

    st.markdown(
        """
        ① Extract Audio  
        ② AI Dialogue Detection  
        ③ Burmese Translation  
        ④ Burmese Voice Generation  
        ⑤ Smart Timing Sync  
        ⑥ Remove Original Audio  
        ⑦ Final Video
        """
    )

    st.markdown("---")

    st.caption(
        "Gemini model is selected automatically from available "
        "stable models."
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "🎥 Movie / Video Upload",
    type=["mp4", "mkv", "mov"],
    help="MP4, MKV or MOV",
)


# ============================================================
# START
# ============================================================

if uploaded_video is not None:

   
