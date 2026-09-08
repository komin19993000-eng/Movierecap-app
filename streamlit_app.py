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


st.set_page_config(
    page_title="Movie Dubbing AI",
    page_icon="🎬",
    layout="wide",
)


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


TEMP_ERROR_WORDS = (
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


def run_cmd(args, timeout=None):
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def format_seconds(seconds):
    seconds = max(0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def get_duration(path):
    result = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(path),
        ],
        timeout=120,
    )

    text = result.stderr or ""

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        text,
    )

    if not match:
        raise RuntimeError("Video duration ကို မဖတ်နိုင်ပါ။")

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))

    return hours * 3600 + minutes * 60 + seconds


def extract_audio(video_path, wav_path):
    result = run_cmd(
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
            str(wav_path),
        ],
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Original audio ထုတ်မရပါ။\n"
            + (result.stderr or "")
        )


# ============================================================
# GEMINI
# ============================================================

@st.cache_resource
def get_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ထဲမှာ GEMINI_API_KEY ထည့်ထားပါ။"
        )

    return genai.Client(api_key=api_key)


def is_temporary_error(error):
    text = str(error).lower()

    return any(
        word.lower() in text
        for word in TEMP_ERROR_WORDS
    )


def get_available_models(client):
    result = []

    try:
        for model in client.models.list():
            name = getattr(model, "name", "")

            if name:
                result.append(
                    name.replace("models/", "")
                )

    except Exception:
        return []

    return result


def choose_models(client):
    available = get_available_models(client)

    if not available:
        return MODEL_CANDIDATES.copy()

    selected = []

    for model in MODEL_CANDIDATES:
        if model in available:
            selected.append(model)

    if selected:
        return selected

    return MODEL_CANDIDATES.copy()


def wait_for_file(client, uploaded):
    name = getattr(uploaded, "name", None)

    if not name:
        return uploaded

    for _ in range(90):
        try:
            current = client.files.get(name=name)

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
        text = text[start:end + 1]

    return text


def normalize_segments(data, duration):
    if not isinstance(data, list):
        raise RuntimeError(
            "Gemini response က JSON array မဟုတ်ပါ။"
        )

    segments = []

    for item in data:

        if not isinstance(item, dict):
            continue

        try:
            start = float(
                item.get("start", 0)
            )

            end = float(
                item.get("end", 0)
            )

        except Exception:
            continue

        text = str(
            item.get("burmese")
            or item.get("translation")
            or item.get("text")
            or ""
        ).strip()

        if not text:
            continue

        start = max(
            0.0,
            min(start, duration),
        )

        end = max(
            0.0,
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
        key=lambda x: (
            x["start"],
            x["end"],
        )
    )

    cleaned = []

    for seg in segments:

        if cleaned:

            previous = cleaned[-1]

            if (
                abs(
                    seg["start"]
                    - previous["start"]
                ) < 0.05
                and
                seg["burmese"]
                == previous["burmese"]
            ):
                previous["end"] = max(
                    previous["end"],
                    seg["end"],
                )
                continue

        cleaned.append(seg)

    return cleaned


def transcribe_translate(
    client,
    wav_path,
    duration,
    status_callback,
):
    try:
        uploaded = client.files.upload(
            file=str(wav_path)
        )

        uploaded = wait_for_file(
            client,
            uploaded,
        )

    except Exception as error:
        raise RuntimeError(
            "Gemini audio upload မအောင်မြင်ပါ။\n"
            + str(error)
        )

    prompt = f"""
You are a professional movie dubbing script editor.

Analyze the uploaded movie audio carefully.

TASK:

1. Detect ALL meaningful spoken dialogue.
2. Keep the dialogue in exact chronological order.
3. Ignore music, sound effects and silence.
4. Give accurate approximate START and END timestamps in seconds.
5. Translate every dialogue line into natural conversational Burmese.
6. Make the Burmese sound like professional movie dubbing.
7. Do NOT translate word-for-word when that sounds unnatural.
8. Preserve meaning, emotion, relationships and names.
9. Do NOT invent dialogue.
10. Keep Burmese dialogue concise enough to fit the original speaking time.
11. Include short meaningful spoken lines.
12. Return ONLY valid JSON.

Audio duration:
{duration:.2f} seconds

Required JSON:

[
  {{
    "start": 12.40,
    "end": 15.80,
    "burmese": "မြန်မာဘာသာပြန်စာကြောင်း"
  }}
]
"""

    models = choose_models(client)

    errors = []

    for model_name in models:

        status_callback(
            f"Gemini model စမ်းနေသည် — {model_name}"
        )

        for attempt in range(2):

            try:

                response = client.models.generate_content(
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

                text = getattr(
                    response,
                    "text",
                    None,
                )

                if not text:
                    raise RuntimeError(
                        "Gemini က response မပြန်ပါ။"
                    )

                parsed = json.loads(
                    clean_json(text)
                )

                segments = normalize_segments(
                    parsed,
                    duration,
                )

                if not segments:
                    raise RuntimeError(
                        "Dialogue segment မတွေ့ပါ။"
                    )

                return segments, model_name

            except Exception as error:

                errors.append(
                    f"{model_name} attempt "
                    f"{attempt + 1}: {error}"
                )

                if (
                    attempt == 0
                    and is_temporary_error(error)
                ):
                    wait = 4 + random.uniform(
                        0,
                        2,
                    )

                    status_callback(
                        f"{model_name} ခဏအလုပ်များနေသည် — "
                        f"{wait:.1f}s နောက် retry..."
                    )

                    time.sleep(wait)

                    continue

                break

    raise RuntimeError(
        "Gemini model အားလုံးနဲ့ "
        "dialogue detection / translation "
        "မအောင်မြင်ပါ။\n\n"
        + "\n".join(errors[-8:])
    )


# ============================================================
# EDGE TTS
# ============================================================

async def create_tts_async(
    text,
    voice,
    output,
):
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
        volume="+0%",
    )

    await communicator.save(
        str(output)
    )


def create_tts(
    text,
    voice,
    output,
):
    asyncio.run(
        create_tts_async(
            text,
            voice,
            output,
        )
    )


def audio_duration(path):
    result = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(path),
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
            "TTS audio duration မဖတ်နိုင်ပါ။"
        )

    return (
        int(match.group(1)) * 3600
        + int(match.group(2)) * 60
        + float(match.group(3))
    )


def atempo_filter(speed):
    speed = max(
        0.5,
        min(float(speed), 3.0),
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

    return ",".join(filters)


def fit_tts_to_slot(
    input_path,
    output_path,
    target_duration,
):
    actual_duration = audio_duration(
        input_path
    )

    if actual_duration <= 0:
        raise RuntimeError(
            "TTS duration မမှန်ပါ။"
        )

    speed = (
        actual_duration
        / max(target_duration, 0.25)
    )

    if 0.97 <= speed <= 1.03:

        result = run_cmd(
            [
                FFMPEG,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(output_path),
            ],
            timeout=120,
        )

    else:

        speed = max(
            0.5,
            min(speed, 3.0),
        )

        result = run_cmd(
            [
                FFMPEG,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-filter:a",
                atempo_filter(speed),
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(output_path),
            ],
            timeout=120,
        )

    if result.returncode != 0:
        raise RuntimeError(
            "TTS timing ပြင်မရပါ။\n"
            + (result.stderr or "")
        )


# ============================================================
# DUB AUDIO
# ============================================================

def create_dubbed_audio(
    segments,
    voice,
    duration,
    workdir,
    callback,
):
    pieces = []

    total = len(segments)

    for index, segment in enumerate(
        segments,
        start=1,
    ):

        start = segment["start"]
        end = segment["end"]

        slot = max(
            0.25,
            end - start,
        )

        raw_audio = (
            workdir
            / f"tts_{index:04d}.mp3"
        )

        fitted_audio = (
            workdir
            / f"fit_{index:04d}.m4a"
        )

        callback(
            f"TTS {index}/{total} — "
            f"{format_seconds(start)} → "
            f"{format_seconds(end)}"
        )

        create_tts(
            segment["burmese"],
            voice,
            raw_audio,
        )

        fit_tts_to_slot(
            raw_audio,
            fitted_audio,
            slot,
        )

        pieces.append(
            (
                start,
                fitted_audio,
            )
        )

    if not pieces:
        raise RuntimeError(
            "TTS audio မထုတ်နိုင်ပါ။"
        )

    output_audio = (
        workdir
        / "burmese_dub.m4a"
    )

    command = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",

        "-f",
        "lavfi",

        "-t",
        f"{duration:.3f}",

        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
    ]

    for _, audio_path in pieces:
        command.extend(
            [
                "-i",
                str(audio_path),
            ]
        )

    filters = []

    filters.append(
        "[0:a]"
        "aresample=48000,"
        f"apad=whole_dur={duration:.3f}"
        "[base]"
    )

    labels = ["[base]"]

    for index, (
        start,
        _,
    ) in enumerate(
        pieces,
        start=1,
    ):

        delay = max(
            0,
            int(round(start * 1000)),
        )

        label = f"[d{index}]"

        filters.append(
            f"[{index}:a]"
            f"aresample=48000,"
            f"adelay={delay}|{delay}"
            f"{label}"
        )

        labels.append(label)

    filters.append(
        "".join(labels)
        + f"amix="
        f"inputs={len(labels)}:"
        f"duration=first:"
        f"dropout_transition=0,"
        f"atrim=duration={duration:.3f},"
        f"alimiter=limit=0.95"
        "[aout]"
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),

            "-map",
            "[aout]",

            "-c:a",
            "aac",

            "-b:a",
            "160k",

            "-movflags",
            "+faststart",

            str(output_audio),
        ]
    )

    result = run_cmd(
        command,
        timeout=900,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Burmese audio mix မအောင်မြင်ပါ။\n"
            + (result.stderr or "")
        )

    return output_audio


# ============================================================
# FINAL VIDEO
# ============================================================

def create_final_video(
    video_path,
    dubbed_audio,
    output_path,
):
    result = run_cmd(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",

            "-i",
            str(video_path),

            "-i",
            str(dubbed_audio),

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

            "-shortest",

            "-map_metadata",
            "-1",

            "-movflags",
            "+faststart",

            str(output_path),
        ],
        timeout=1800,
    )

    if result.returncode == 0:
        return

    # Video stream copy မရရင် H.264 နဲ့ fallback re-encode
    result2 = run_cmd(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",

            "-i",
            str(video_path),

            "-i",
            str(dubbed_audio),

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

            "-shortest",

            "-movflags",
            "+faststart",

            str(output_path),
        ],
        timeout=3600,
    )

    if result2.returncode != 0:
        raise RuntimeError(
            "Final video export မအောင်မြင်ပါ။\n"
            + (
                result2.stderr
                or result.stderr
                or ""
            )
        )


# ============================================================
# UI
# ============================================================

st.title("🎬 Movie Dubbing AI")

st.caption(
    "Video → Dialogue → Burmese Translation → "
    "Burmese Voice → Final Video"
)


with st.sidebar:

    st.subheader("🎙️ Burmese Voice")

    voice_name = st.selectbox(
        "အသံရွေးပါ",
        list(VOICE_MAP.keys()),
    )

    selected_voice = VOICE_MAP[
        voice_name
    ]

    st.divider()

    st.caption(
        "Gemini model ကို အလိုအလျောက် "
        "ရွေးပြီး retry / fallback လုပ်ပေးပါမယ်။"
    )


uploaded_video = st.file_uploader(
    "🎥 Movie / Video တင်ပါ",
    type=[
        "mp4",
        "mkv",
        "mov",
        "avi",
        "webm",
    ],
)


if uploaded_video:

    st.video(uploaded_video)

    start = st.button(
        "🚀 START DUBBING",
        type="primary",
        use_container_width=True,
    )

    if start:

        progress = st.progress(0)
        status = st.empty()
        eta = st.empty()

        started_at = time.time()

        def update(
            percent,
            message,
        ):
            percent = max(
                0,
                min(100, percent),
            )

            progress.progress(
                int(percent)
            )

            status.info(message)

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

                eta.caption(
                    f"Progress: {percent:.0f}% "
                    f"• ETA: {int(remaining)} sec"
                )

        try:

            with tempfile.TemporaryDirectory(
                prefix="movie_dub_"
            ) as temp:

                workdir = Path(temp)

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
                # 1
                # ------------------------------------------------

                update(
                    5,
                    "1/5 Video စစ်ဆေးနေသည်...",
                )

                duration = get_duration(
                    input_path
                )

                # ------------------------------------------------
                # 2
                # ------------------------------------------------

                update(
                    12,
                    "2/5 Original audio ထုတ်နေသည်...",
                )

                wav_path = (
                    workdir
                    / "source_audio.wav"
                )

                extract_audio(
                    input_path,
                    wav_path,
                )

                # ------------------------------------------------
                # 3
                # ------------------------------------------------

                update(
                    18,
                    "3/5 Gemini AI နဲ့ dialogue "
                    "စစ်နေသည်...",
                )

                client = get_gemini_client()

                def gemini_status(message):
                    update(
                        20,
                        "3/5 " + message,
                    )

                segments, model_used = (
                    transcribe_translate(
                        client,
                        wav_path,
                        duration,
                        gemini_status,
                    )
                )

                update(
                    45,
                    f"3/5 Dialogue "
                    f"{len(segments)} ခု ရပြီ "
                    f"• {model_used}",
                )

                # ------------------------------------------------
                # 4
                # ------------------------------------------------

                tts_count = [0]

                def tts_status(message):

                    tts_count[0] += 1

                    percent = (
                        48
                        + (
                            tts_count[0]
                            / max(
                                1,
                                len(segments),
                            )
                        )
                        * 34
                    )

                    update(
                        percent,
                        "4/5 " + message,
                    )

                dubbed_audio = (
                    create_dubbed_audio(
                        segments,
                        selected_voice,
                        duration,
                        workdir,
                        tts_status,
                    )
                )

                # ------------------------------------------------
                # 5
                # ------------------------------------------------

                update(
                    88,
                    "5/5 Burmese voice နဲ့ "
                    "video ပေါင်းနေသည်...",
                )

                output_path = (
                    workdir
                    / "Movie_Dubbed_Burmese.mp4"
                )

                create_final_video(
                    input_path,
                    dubbed_audio,
                    output_path,
                )

                update(
                    100,
                    "✅ ပြီးပါပြီ",
                )

                st.success(
                    f"အောင်မြင်ပါပြီ။ "
                    f"Dialogue {len(segments)} ခု "
                    f"• Model: {model_used}"
                )

                with open(
                    output_path,
                    "rb",
                ) as file:

                    video_data = file.read()

                st.download_button(
                    "⬇️ Download Final Video",
                    data=video_data,
                    file_name=(
                        "Movie_Dubbed_Burmese.mp4"
                    ),
                    mime="video/mp4",
                    use_container_width=True,
                )

        except Exception as error:

            progress.progress(0)

            status.error(
                "❌ Processing မအောင်မြင်ပါ"
            )

            eta.empty()

            st.exception(error)

else:

    st.info(
        "Video တင်ပြီး "
        "START DUBBING ကိုနှိပ်ပါ။"
    )
