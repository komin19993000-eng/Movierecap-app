import os
import re
import json
import time
import asyncio
import random
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
import edge_tts
import imageio_ffmpeg


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Movie Dubbing AI",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Movie Dubbing AI")
st.caption(
    "Video → AI Dialogue → Burmese Translation → "
    "Burmese Voice → Preview → Download"
)


# ============================================================
# SETTINGS
# ============================================================

VOICE_MAP = {
    "သီဟ (အမျိုးသား)": "my-MM-ThihaNeural",
    "နီလာ (အမျိုးသမီး)": "my-MM-NilarNeural",
}

MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

TEMP_ERRORS = (
    "503",
    "500",
    "502",
    "504",
    "429",
    "timeout",
    "timed out",
    "unavailable",
    "overloaded",
    "high demand",
    "temporarily",
    "resource exhausted",
)


# ============================================================
# FFMPEG
# ============================================================

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def run_command(command, timeout=600):
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
    )


def get_duration(file_path):
    result = run_command(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(file_path),
        ],
        timeout=120,
    )

    text = result.stderr or ""

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        text,
    )

    if not match:
        raise RuntimeError(
            "Video duration ကို ဖတ်မရပါ။"
        )

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )


def extract_audio(video_path, audio_path):
    result = run_command(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
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
        timeout=900,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Original audio ထုတ်မရပါ။\n\n"
            + (result.stderr or "")
        )

    if not audio_path.exists():
        raise RuntimeError(
            "Audio file မထွက်လာပါ။"
        )

    if audio_path.stat().st_size < 1000:
        raise RuntimeError(
            "Extracted audio file က အလွတ်ဖြစ်နေပါတယ်။"
        )


# ============================================================
# GEMINI
# ============================================================

@st.cache_resource
def get_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.environ.get(
            "GEMINI_API_KEY",
            "",
        )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return genai.Client(
        api_key=api_key
    )


def is_temporary_error(error):
    text = str(error).lower()

    return any(
        word.lower() in text
        for word in TEMP_ERRORS
    )


def get_available_models(client):
    try:
        models = client.models.list()

        result = []

        for model in models:
            name = getattr(
                model,
                "name",
                "",
            )

            if name:
                result.append(
                    name.replace(
                        "models/",
                        "",
                    )
                )

        return result

    except Exception:
        return []


def select_models(client):
    available = get_available_models(
        client
    )

    if not available:
        return MODEL_CANDIDATES.copy()

    selected = [
        model
        for model in MODEL_CANDIDATES
        if model in available
    ]

    return (
        selected
        if selected
        else MODEL_CANDIDATES.copy()
    )


def wait_for_uploaded_file(
    client,
    uploaded,
):
    name = getattr(
        uploaded,
        "name",
        None,
    )

    if not name:
        return uploaded

    for _ in range(90):

        try:
            current = client.files.get(
                name=name
            )

            state = getattr(
                current,
                "state",
                None,
            )

            state_name = str(
                getattr(
                    state,
                    "name",
                    state,
                )
            ).upper()

            if "PROCESSING" not in state_name:
                return current

            time.sleep(2)

        except Exception:
            return uploaded

    return uploaded


def clean_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.I,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    start = text.find("[")
    end = text.rfind("]")

    if start >= 0 and end > start:
        text = text[
            start:end + 1
        ]

    return text


def normalize_segments(
    data,
    duration,
):
    if not isinstance(data, list):
        raise RuntimeError(
            "Gemini response က JSON list မဟုတ်ပါ။"
        )

    segments = []

    for item in data:

        if not isinstance(
            item,
            dict,
        ):
            continue

        try:
            start = float(
                item.get(
                    "start",
                    0,
                )
            )

            end = float(
                item.get(
                    "end",
                    0,
                )
            )

        except Exception:
            continue

        text = str(
            item.get(
                "burmese",
                "",
            )
        ).strip()

        if not text:
            continue

        start = max(
            0,
            min(start, duration),
        )

        end = max(
            0,
            min(end, duration),
        )

        if end <= start:
            continue

        if end - start < 0.20:
            continue

        segments.append(
            {
                "start": start,
                "end": end,
                "burmese": text,
            }
        )

    segments.sort(
        key=lambda x: x["start"]
    )

    # Duplicate dialogue removal
    cleaned = []

    for segment in segments:

        if cleaned:

            previous = cleaned[-1]

            if (
                abs(
                    segment["start"]
                    - previous["start"]
                ) < 0.05
                and
                segment["burmese"]
                == previous["burmese"]
            ):
                previous["end"] = max(
                    previous["end"],
                    segment["end"],
                )
                continue

        cleaned.append(segment)

    return cleaned


def analyze_movie_audio(
    client,
    audio_path,
    duration,
    status_callback,
):
    try:

        uploaded = client.files.upload(
            file=str(audio_path)
        )

        uploaded = wait_for_uploaded_file(
            client,
            uploaded,
        )

    except Exception as error:
        raise RuntimeError(
            "Gemini audio upload မအောင်မြင်ပါ။\n\n"
            + str(error)
        )

    prompt = f"""
You are a professional movie dubbing editor.

Analyze the uploaded movie audio.

Find ALL meaningful spoken dialogue.

Requirements:

1. Detect every meaningful spoken line.
2. Do not invent dialogue.
3. Keep chronological order.
4. Give approximate START and END timestamps in seconds.
5. Translate into natural conversational Burmese.
6. Burmese must sound like professional movie dubbing.
7. Preserve emotion and meaning.
8. Keep names and relationships correct.
9. Do not include music or sound effects.
10. Keep Burmese sentences reasonably concise so they can fit the original timing.
11. Do not merge unrelated dialogue lines.
12. Return ONLY valid JSON.

Audio duration:
{duration:.2f} seconds

JSON format:

[
  {{
    "start": 10.25,
    "end": 13.80,
    "burmese": "မြန်မာဘာသာပြန်"
  }}
]
"""

    models = select_models(
        client
    )

    errors = []

    for model_name in models:

        status_callback(
            f"AI model စမ်းနေသည် — {model_name}"
        )

        for attempt in range(2):

            try:

                response = (
                    client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Part.from_uri(
                                file_uri=uploaded.uri,
                                mime_type=(
                                    uploaded.mime_type
                                    or "audio/wav"
                                ),
                            ),
                            prompt,
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0.15,
                        ),
                    )
                )

                text = getattr(
                    response,
                    "text",
                    None,
                )

                if not text:
                    raise RuntimeError(
                        "Gemini response မရှိပါ။"
                    )

                data = json.loads(
                    clean_json(text)
                )

                segments = normalize_segments(
                    data,
                    duration,
                )

                if not segments:
                    raise RuntimeError(
                        "Dialogue မတွေ့ပါ။"
                    )

                return (
                    segments,
                    model_name,
                )

            except Exception as error:

                errors.append(
                    f"{model_name} "
                    f"attempt {attempt + 1}: "
                    f"{error}"
                )

                if (
                    attempt == 0
                    and is_temporary_error(
                        error
                    )
                ):

                    wait = (
                        4
                        + random.uniform(
                            0,
                            2,
                        )
                    )

                    status_callback(
                        f"{model_name} "
                        f"ခဏအလုပ်များနေသည်။ "
                        f"{wait:.1f}s နောက် retry..."
                    )

                    time.sleep(
                        wait
                    )

                    continue

                break

    raise RuntimeError(
        "Gemini model အားလုံးနဲ့ "
        "dialogue translation မအောင်မြင်ပါ။\n\n"
        + "\n".join(
            errors[-8:]
        )
    )


# ============================================================
# TTS
# ============================================================

async def generate_tts_async(
    text,
    voice,
    output_path,
):
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
        volume="+0%",
    )

    await communicator.save(
        str(output_path)
    )


def generate_tts(
    text,
    voice,
    output_path,
):
    asyncio.run(
        generate_tts_async(
            text,
            voice,
            output_path,
        )
    )

    if not output_path.exists():
        raise RuntimeError(
            "TTS file မထွက်လာပါ။"
        )

    if output_path.stat().st_size < 1000:
        raise RuntimeError(
            "TTS file အလွတ်ဖြစ်နေပါတယ်။"
        )


def get_audio_duration(
    audio_path
):
    return get_duration(
        audio_path
    )


def build_atempo_filter(
    speed
):
    speed = max(
        0.5,
        min(
            float(speed),
            3.0,
        ),
    )

    filters = []

    while speed > 2.0:
        filters.append(
            "atempo=2.0"
        )
        speed /= 2.0

    while speed < 0.5:
        filters.append(
            "atempo=0.5"
        )
        speed /= 0.5

    filters.append(
        f"atempo={speed:.6f}"
    )

    return ",".join(
        filters
    )


def fit_audio_to_slot(
    input_audio,
    output_audio,
    slot_duration,
):
    original_duration = (
        get_audio_duration(
            input_audio
        )
    )

    if original_duration <= 0:
        raise RuntimeError(
            "TTS duration မမှန်ပါ။"
        )

    speed = (
        original_duration
        / max(
            slot_duration,
            0.25,
        )
    )

    speed = max(
        0.5,
        min(
            speed,
            3.0,
        ),
    )

    result = run_command(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_audio),
            "-filter:a",
            build_atempo_filter(
                speed
            ),
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(output_audio),
        ],
        timeout=180,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "TTS timing ပြင်မရပါ။\n\n"
            + (
                result.stderr
                or ""
            )
        )

    if not output_audio.exists():
        raise RuntimeError(
            "Fitted TTS file မထွက်ပါ။"
        )


# ============================================================
# CREATE BURMESE AUDIO
# ============================================================

def create_burmese_audio(
    segments,
    voice,
    duration,
    workdir,
    callback,
):
    audio_files = []

    total = len(
        segments
    )

    for index, segment in enumerate(
        segments,
        start=1,
    ):

        start = float(
            segment["start"]
        )

        end = float(
            segment["end"]
        )

        slot = max(
            0.25,
            end - start,
        )

        text = segment[
            "burmese"
        ]

        raw_tts = (
            workdir
            / f"tts_{index:04d}.mp3"
        )

        fitted = (
            workdir
            / f"fitted_{index:04d}.m4a"
        )

        callback(
            index,
            total,
            f"TTS {index}/{total}"
        )

        generate_tts(
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
                "start": start,
                "file": fitted,
            }
        )

    if not audio_files:
        raise RuntimeError(
            "Burmese TTS audio မရှိပါ။"
        )

    # --------------------------------------------------------
    # Build FFmpeg command
    # --------------------------------------------------------

    command = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    for item in audio_files:

        command.extend(
            [
                "-i",
                str(
                    item["file"]
                ),
            ]
        )

    filters = []

    labels = []

    for index, item in enumerate(
        audio_files
    ):

        delay = int(
            round(
                item["start"]
                * 1000
            )
        )

        label = f"a{index}"

        filters.append(
            f"[{index}:a]"
            f"aresample=48000,"
            f"adelay={delay}|{delay},"
            f"apad"
            f"[{label}]"
        )

        labels.append(
            f"[{label}]"
        )

    mixed = "[mixed]"

    filters.append(
        "".join(labels)
        + f"amix="
        f"inputs={len(labels)}:"
        f"duration=longest:"
        f"dropout_transition=0"
        f"{mixed}"
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(
                filters
            ),
            "-map",
            mixed,
            "-t",
            f"{duration:.3f}",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
        ]
    )

    output_audio = (
        workdir
        / "burmese_audio.m4a"
    )

    command.append(
        str(output_audio)
    )

    result = run_command(
        command,
        timeout=1200,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Burmese audio mixing မအောင်မြင်ပါ။\n\n"
            + (
                result.stderr
                or ""
            )
        )

    if not output_audio.exists():
        raise RuntimeError(
            "Burmese audio file မထွက်လာပါ။"
        )

    if output_audio.stat().st_size < 5000:
        raise RuntimeError(
            "Burmese audio file အလွတ်ဖြစ်နေပါတယ်။"
        )

    return output_audio


# ============================================================
# FINAL VIDEO
# ============================================================

def export_final_video(
    video_path,
    audio_path,
    output_path,
):
    result = run_command(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",

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
            "160k",

            "-ar",
            "48000",

            "-ac",
            "2",

            "-t",
            "0",

            "-movflags",
            "+faststart",

            str(output_path),
        ],
        timeout=1800,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # The first export intentionally retries with correct
    # video duration below.
    # --------------------------------------------------------

    if output_path.exists():
        try:
            if output_path.stat().st_size > 10000:
                return
        except Exception:
            pass

    # Fallback export
    video_duration = get_duration(
        video_path
    )

    result = run_command(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",

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
            "160k",

            "-ar",
            "48000",

            "-ac",
            "2",

            "-t",
            f"{video_duration:.3f}",

            "-movflags",
            "+faststart",

            str(output_path),
        ],
        timeout=1800,
    )

    if result.returncode != 0:
        # Final fallback: re-encode video
        result = run_command(
            [
                FFMPEG,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",

                "-i",
                str(video_path),

                "-i",
                str(audio_path),

                "-map",
                "0:v:0",
                "-map",
                "1:a:0",

                "-c:v",
                "libx264",

                "-preset",
                "veryfast",

                "-crf",
                "23",

                "-pix_fmt",
                "yuv420p",

                "-c:a",
                "aac",

                "-b:a",
                "160k",

                "-ar",
                "48000",

                "-ac",
                "2",

                "-t",
                f"{video_duration:.3f}",

                "-movflags",
                "+faststart",

                str(output_path),
            ],
            timeout=3600,
        )

    if result.returncode != 0:
        raise RuntimeError(
            "Final video export မအောင်မြင်ပါ။\n\n"
            + (
                result.stderr
                or ""
            )
        )


# ============================================================
# VALIDATE FINAL VIDEO
# ============================================================

def validate_final_video(
    video_path
):
    if not video_path.exists():
        return False, "Final video file မရှိပါ။"

    if video_path.stat().st_size < 10000:
        return False, "Final video file အလွတ်နီးပါးဖြစ်နေပါတယ်။"

    # Check video
    video_check = run_command(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        timeout=180,
    )

    if video_check.returncode != 0:
        return False, (
            "Video track မမှန်ပါ။\n"
            + (
                video_check.stderr
                or ""
            )
        )

    # Check audio
    audio_check = run_command(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-t",
            "1",
            "-f",
            "null",
            "-",
        ],
        timeout=180,
    )

    if audio_check.returncode != 0:
        return False, (
            "❌ Final video ထဲမှာ Audio track မရှိပါ။"
        )

    # Check duration
    try:
        duration = get_duration(
            video_path
        )

        if duration <= 0:
            return False, (
                "Video duration မမှန်ပါ။"
            )

    except Exception as error:
        return False, str(error)

    return True, (
        f"Video OK • Audio OK • "
        f"Duration {duration:.1f}s"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader(
        "🎙️ Burmese Voice"
    )

    voice_name = st.selectbox(
        "အသံရွေးပါ",
        list(
            VOICE_MAP.keys()
        ),
    )

    selected_voice = VOICE_MAP[
        voice_name
    ]

    st.divider()

    st.caption(
        "Final Video ကို Download မလုပ်ခင် "
        "ဒီ App ထဲမှာ Preview ကြည့်နိုင်ပါတယ်။"
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "🎥 Video တင်ပါ",
    type=[
        "mp4",
        "mkv",
        "mov",
        "avi",
        "webm",
    ],
)


if uploaded_video:

    st.video(
        uploaded_video
    )

    start_button = st.button(
        "🚀 START DUBBING",
        type="primary",
        use_container_width=True,
    )

    if start_button:

        progress = st.progress(
            0
        )

        status = st.empty()

        eta_box = st.empty()

        started_at = time.time()

        def update(
            percent,
            message,
        ):
            percent = max(
                0,
                min(
                    100,
                    percent,
                ),
            )

            progress.progress(
                int(percent)
            )

            status.info(
                message
            )

            elapsed = (
                time.time()
                - started_at
            )

            if percent > 1:

                estimated = (
                    elapsed
                    * 100
                    / percent
                )

                remaining = max(
                    0,
                    estimated
                    - elapsed,
                )

                eta_box.caption(
                    f"Progress: "
                    f"{percent:.0f}% "
                    f"• ETA: "
                    f"{int(remaining)} sec"
                )

        try:

            with tempfile.TemporaryDirectory(
                prefix="movie_dubbing_"
            ) as temp_dir:

                workdir = Path(
                    temp_dir
                )

                # ------------------------------------------------
                # Save input
                # ------------------------------------------------

                input_path = (
                    workdir
                    / uploaded_video.name
                )

                with open(
                    input_path,
                    "wb",
                ) as file:

                    file.write(
                        uploaded_video.getbuffer()
                    )

                # ------------------------------------------------
                # 1. VIDEO CHECK
                # ------------------------------------------------

                update(
                    5,
                    "1/6 Video စစ်ဆေးနေသည်..."
                )

                duration = get_duration(
                    input_path
                )

                # ------------------------------------------------
                # 2. AUDIO
                # ------------------------------------------------

                update(
                    10,
                    "2/6 Original audio ထုတ်နေသည်..."
                )

                source_audio = (
                    workdir
                    / "source_audio.wav"
                )

                extract_audio(
                    input_path,
                    source_audio,
                )

                # ------------------------------------------------
                # 3. GEMINI
                # ------------------------------------------------

                update(
                    18,
                    "3/6 Gemini AI "
                    "dialogue စစ်နေသည်..."
                )

                client = get_client()

                def ai_status(message):
                    update(
                        20,
                        "3/6 " + message
                    )

                segments, model_used = (
                    analyze_movie_audio(
                        client,
                        source_audio,
                        duration,
                        ai_status,
                    )
                )

                update(
                    45,
                    f"3/6 Dialogue "
                    f"{len(segments)} ခုရပြီ "
                    f"• {model_used}"
                )

                # ------------------------------------------------
                # 4. TTS
                # ------------------------------------------------

                def tts_status(
                    index,
                    total,
                    message,
                ):
                    percent = (
                        48
                        + (
                            index
                            / max(
                                total,
                                1,
                            )
                        )
                        * 32
                    )

                    update(
                        percent,
                        "4/6 " + message
                    )

                burmese_audio = (
                    create_burmese_audio(
                        segments,
                        selected_voice,
                        duration,
                        workdir,
                        tts_status,
                    )
                )

                # ------------------------------------------------
                # Check Burmese audio before mux
                # ------------------------------------------------

                update(
                    82,
                    "Burmese audio "
                    "တကယ်ထွက်မထွက် စစ်နေသည်..."
                )

                audio_test = run_command(
                    [
                        FFMPEG,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(
                            burmese_audio
                        ),
                        "-map",
                        "0:a:0",
                        "-t",
                        "1",
                        "-f",
                        "null",
                        "-",
                    ],
                    timeout=180,
                )

                if audio_test.returncode != 0:
                    raise RuntimeError(
                        "Burmese TTS audio ကို "
                        "စစ်တဲ့အခါ မအောင်မြင်ပါ။"
                    )

                # ------------------------------------------------
                # 5. FINAL VIDEO
                # ------------------------------------------------

                update(
                    88,
                    "5/6 Final video "
                    "နဲ့ Burmese audio ပေါင်းနေသည်..."
                )

                final_path = (
                    workdir
                    / "Movie_Dubbed_Burmese.mp4"
                )

                export_final_video(
                    input_path,
                    burmese_audio,
                    final_path,
                )

                # ------------------------------------------------
                # 6. VALIDATE
                # ------------------------------------------------

                update(
                    96,
                    "6/6 Final video "
                    "ကို စစ်ဆေးနေသည်..."
                )

                valid, validation_message = (
                    validate_final_video(
                        final_path
                    )
                )

                if not valid:
                    raise RuntimeError(
                        validation_message
                    )

                update(
                    100,
                    "✅ Final Video အဆင်ပြေပါပြီ"
                )

                st.success(
                    "🎉 Final Video အောင်မြင်ပါတယ်။"
                )

                st.info(
                    "👇 အရင်ဆုံး ဒီနေရာမှာ Video ကို "
                    "ကြည့်ပြီး အသံပါ/မပါ စစ်ပါ။ "
                    "အဆင်ပြေမှ Download လုပ်ပါ။"
                )

                # =================================================
                # FINAL PREVIEW
                # =================================================

                st.subheader(
                    "🎬 Final Video Preview"
                )

                st.video(
                    str(final_path)
                )

                st.success(
                    "🔊 Audio Track: OK\n\n"
                    + validation_message
                )

                # =================================================
                # DOWNLOAD
                # =================================================

                with open(
                    final_path,
                    "rb",
                ) as file:

                    final_video_bytes = (
                        file.read()
                    )

                st.download_button(
                    "⬇️ Download Final Video",
                    data=final_video_bytes,
                    file_name=(
                        "Movie_Dubbed_Burmese.mp4"
                    ),
                    mime="video/mp4",
                    use_container_width=True,
                )

        except Exception as error:

            progress.progress(
                0
            )

            eta_box.empty()

            status.error(
                "❌ Processing မအောင်မြင်ပါ။"
            )

            st.error(
                str(error)
            )

else:

    st.info(
        "Video တင်ပြီး "
        "START DUBBING ကိုနှိပ်ပါ။"
    )
