import os
import json
import time
import asyncio
import random
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
)

st.title("🎬 AI Movie Recap Automator")
st.caption(
    "Natural Burmese Translation • Burmese AI Voice • Smart Dialogue Timing"
)


# ============================================================
# SETTINGS
# ============================================================

VOICE_MAP = {
    "အမျိုးသား (သီဟ)": "my-MM-ThihaNeural",
    "အမျိုးသမီး (နီလာ)": "my-MM-NilarNeural",
}

# Current stable models first.
MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    voice_name = st.selectbox(
        "🎙️ Burmese Voice",
        list(VOICE_MAP.keys()),
    )

    st.divider()

    st.markdown(
        """
        **Pipeline**

        1. 🎥 Upload Video
        2. 🎵 Extract Audio
        3. 🤖 AI Dialogue Detection
        4. 🇲🇲 Burmese Translation
        5. 🎙️ Burmese TTS
        6. ⏱️ Smart Timing
        7. 🎬 Final Video
        """
    )


# ============================================================
# COMMAND
# ============================================================

def run_cmd(command, timeout=3600):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        error = result.stderr.strip()

        if len(error) > 6000:
            error = error[-6000:]

        raise RuntimeError(
            error or "Command failed."
        )

    return result.stdout


# ============================================================
# VIDEO DURATION
# ============================================================

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


# ============================================================
# JSON
# ============================================================

def parse_json_response(text):
    if not text:
        raise RuntimeError(
            "Gemini response is empty."
        )

    text = text.strip()

    if "```json" in text:
        text = text.replace(
            "```json",
            "",
        )

    text = text.replace(
        "```",
        "",
    ).strip()

    start = text.find("[")
    end = text.rfind("]")

    if start < 0 or end <= start:
        raise RuntimeError(
            "Gemini returned invalid JSON."
        )

    json_text = text[
        start:end + 1
    ]

    return json.loads(json_text)


# ============================================================
# GEMINI MODELS
# ============================================================

def get_available_models(client):
    try:
        available = []

        for model in client.models.list():

            name = getattr(
                model,
                "name",
                "",
            )

            if not name:
                continue

            if name.startswith("models/"):
                name = name[
                    len("models/"):
                ]

            available.append(name)

        return available

    except Exception:
        return []


def is_temporary_error(error_text):
    error_text = str(
        error_text
    ).lower()

    temporary_words = [
        "503",
        "unavailable",
        "high demand",
        "overloaded",
        "temporarily",
        "internal server error",
        "500",
        "502",
        "504",
        "timeout",
        "timed out",
    ]

    return any(
        word in error_text
        for word in temporary_words
    )


def generate_with_retry(
    client,
    contents,
    status_callback=None,
):
    """
    Gemini generation with:
    - model fallback
    - exponential backoff
    - jitter
    - limited retries
    """

    available = get_available_models(
        client
    )

    if available:
        candidates = [
            model
            for model in MODEL_CANDIDATES
            if model in available
        ]

        if not candidates:
            candidates = MODEL_CANDIDATES

    else:
        candidates = MODEL_CANDIDATES

    errors = []

    max_retries_per_model = 3

    for model_index, model in enumerate(
        candidates
    ):

        for attempt in range(
            max_retries_per_model
        ):

            try:

                if status_callback:
                    status_callback(
                        f"🤖 Gemini "
                        f"{model} ကို စမ်းနေသည်... "
                        f"({attempt + 1}/"
                        f"{max_retries_per_model})"
                    )

                response = (
                    client.models.generate_content(
                        model=model,
                        contents=contents,
                    )
                )

                text = getattr(
                    response,
                    "text",
                    None,
                )

                if text:
                    return model, text

                errors.append(
                    f"{model}: empty response"
                )

            except Exception as error:

                error_text = str(error)

                errors.append(
                    f"{model} "
                    f"attempt {attempt + 1}: "
                    f"{error_text}"
                )

                if not is_temporary_error(
                    error_text
                ):
                    break

                # Exponential backoff:
                # 3s -> 6s -> 12s
                delay = (
                    3 * (2 ** attempt)
                )

                # Small random jitter.
                delay += random.uniform(
                    0,
                    2,
                )

                if status_callback:
                    status_callback(
                        f"⏳ {model} ခဏအကြာ "
                        f"ပြန်စမ်းမည်... "
                        f"{delay:.0f}s"
                    )

                time.sleep(delay)

        if status_callback:
            status_callback(
                f"➡️ {model} မရသေးပါ။ "
                f"နောက် model ကို ပြောင်းနေသည်..."
            )

    error_message = (
        "Gemini model အားလုံး failed.\n\n"
        + "\n".join(
            errors[-20:]
        )
    )

    raise RuntimeError(
        error_message
    )


# ============================================================
# EXTRACT AUDIO
# ============================================================

def extract_audio(
    video_path,
    audio_path,
):
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


# ============================================================
# GEMINI FILE UPLOAD
# ============================================================

def upload_audio(
    client,
    audio_path,
):
    return client.files.upload(
        file=str(audio_path)
    )


# ============================================================
# AI TRANSCRIPTION + TRANSLATION
# ============================================================

def analyze_audio(
    client,
    uploaded_audio,
    status_callback=None,
):
    prompt = """
You are a professional Myanmar movie recap narrator.

Listen to the entire uploaded audio.

Detect every meaningful spoken dialogue.

For every dialogue:

1. Find the exact approximate start time.
2. Find the exact approximate end time.
3. Understand the context.
4. Translate it into natural spoken Burmese.
5. Make the Burmese sound like a professional movie narrator.
6. Preserve emotion and meaning.
7. Do not translate word-for-word when that sounds unnatural.
8. Do not invent dialogue.
9. Do not skip important spoken dialogue.
10. Ignore music and sound effects.
11. Keep chronological order.

IMPORTANT:

Return ONLY valid JSON.

Format:

[
  {
    "start": 0.00,
    "end": 3.50,
    "text": "မြန်မာဘာသာဖြင့် သဘာဝကျကျ ပြန်ဆိုထားသော စကား"
  }
]

Rules:

- start and end are seconds.
- Do not write "s".
- Do not use markdown.
- Do not add explanations.
- Use Burmese script.
"""

    model, response = generate_with_retry(
        client,
        [
            uploaded_audio,
            prompt,
        ],
        status_callback,
    )

    segments = parse_json_response(
        response
    )

    return model, segments


# ============================================================
# CLEAN SEGMENTS
# ============================================================

def clean_segments(
    raw_segments,
    total_duration,
):
    cleaned = []

    for item in raw_segments:

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

            text = str(
                item.get(
                    "text",
                    "",
                )
            ).strip()

        except Exception:
            continue

        start = max(
            0.0,
            min(
                start,
                total_duration,
            ),
        )

        end = max(
            0.0,
            min(
                end,
                total_duration,
            ),
        )

        if (
            end <= start
            or not text
        ):
            continue

        cleaned.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    cleaned.sort(
        key=lambda x: x["start"]
    )

    return cleaned


# ============================================================
# TTS
# ============================================================

def create_tts(
    text,
    voice,
    output_path,
):
    async def generate():
        communicator = (
            edge_tts.Communicate(
                text=text,
                voice=voice,
                rate="+0%",
                volume="+0%",
                pitch="+0Hz",
            )
        )

        await communicator.save(
            str(output_path)
        )

    asyncio.run(
        generate()
    )


# ============================================================
# AUDIO SPEED
# ============================================================

def build_atempo_filter(
    speed
):
    speed = max(
        0.5,
        min(
            2.0,
            speed,
        ),
    )

    return (
        f"atempo={speed:.6f}"
    )


def fit_tts_to_slot(
    input_audio,
    output_audio,
    target_duration,
):
    source_duration = get_duration(
        input_audio
    )

    if source_duration <= 0:
        raise RuntimeError(
            "Invalid TTS duration."
        )

    target_duration = max(
        0.1,
        target_duration,
    )

    speed = (
        source_duration /
        target_duration
    )

    # Prevent unnatural extreme speed.
    speed = max(
        0.5,
        min(
            2.0,
            speed,
        ),
    )

    filter_value = (
        build_atempo_filter(
            speed
        )
    )

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
# BUILD CONTINUOUS BURMESE AUDIO
# ============================================================

def build_burmese_audio(
    segments,
    voice,
    total_duration,
    work_dir,
    progress_callback=None,
):
    silence_path = (
        work_dir /
        "silence.m4a"
    )

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
            str(silence_path),
        ],
        timeout=300,
    )

    tts_files = []

    total = len(
        segments
    )

    for index, segment in enumerate(
        segments
    ):

        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        slot_duration = (
            end - start
        )

        raw_tts = (
            work_dir /
            f"tts_{index:05d}.mp3"
        )

        fitted_tts = (
            work_dir /
            f"fit_{index:05d}.m4a"
        )

        create_tts(
            text,
            voice,
            raw_tts,
        )

        fit_tts_to_slot(
            raw_tts,
            fitted_tts,
            slot_duration,
        )

        tts_files.append(
            (
                fitted_tts,
                start,
            )
        )

        if progress_callback:
            progress_callback(
                0.40
                +
                (
                    0.35
                    *
                    (
                        (index + 1)
                        /
                        max(
                            1,
                            total,
                        )
                    )
                ),
                (
                    f"🎙️ Burmese Voice "
                    f"{index + 1}/{total}"
                ),
            )

    inputs = [
        "-i",
        str(silence_path),
    ]

    filters = [
        "[0:a]aresample=48000[base]"
    ]

    mix_inputs = [
        "[base]"
    ]

    for index, (
        audio_path,
        start,
    ) in enumerate(
        tts_files,
        start=1,
    ):

        delay = max(
            0,
            int(
                start * 1000
            ),
        )

        inputs.extend(
            [
                "-i",
                str(audio_path),
            ]
        )

        filters.append(
            f"[{index}:a]"
            f"aresample=48000,"
            f"adelay={delay}|{delay}"
            f"[a{index}]"
        )

        mix_inputs.append(
            f"[a{index}]"
        )

    filters.append(
        "".join(mix_inputs)
        +
        f"amix="
        f"inputs={len(mix_inputs)}:"
        f"duration=first:"
        f"dropout_transition=0:"
        f"normalize=0,"
        f"atrim=0:{total_duration:.3f},"
        f"asetpts=PTS-STARTPTS"
        f"[out]"
    )

    output_audio = (
        work_dir /
        "burmese_full.m4a"
    )

    run_cmd(
        [
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
        ],
        timeout=3600,
    )

    return output_audio


# ============================================================
# FINAL VIDEO
# ============================================================

def merge_video_audio(
    video_path,
    audio_path,
    output_path,
):
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
# UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "🎥 Upload Movie / Video",
    type=[
        "mp4",
        "mkv",
        "mov",
    ],
)


# ============================================================
# MAIN
# ============================================================

if uploaded_video is not None:

    file_size = (
        uploaded_video.size
        /
        (1024 * 1024)
    )

    st.info(
        f"📁 {uploaded_video.name} "
        f"• {file_size:.1f} MB"
    )

    start = st.button(
        "🚀 START AI DUBBING",
        type="primary",
        use_container_width=True,
    )

    if start:

        api_key = (
            st.secrets.get(
                "GEMINI_API_KEY",
                None,
            )
        )

        if not api_key:
            api_key = os.environ.get(
                "GEMINI_API_KEY"
            )

        if not api_key:

            st.error(
                "❌ GEMINI_API_KEY မတွေ့ပါ။"
            )

            st.stop()

        progress = st.progress(
            0
        )

        status = st.empty()

        def update(
            value,
            message,
        ):
            progress.progress(
                int(
                    value * 100
                )
            )

            status.write(
                f"**{message}**"
            )

        try:

            with tempfile.TemporaryDirectory() as temp:

                work_dir = Path(
                    temp
                )

                suffix = (
                    Path(
                        uploaded_video.name
                    ).suffix
                    or ".mp4"
                )

                video_path = (
                    work_dir /
                    f"input{suffix}"
                )

                audio_path = (
                    work_dir /
                    "audio.wav"
                )

                output_path = (
                    work_dir /
                    "Myanmar_Dub.mp4"
                )

                # ------------------------------------------------
                # SAVE VIDEO
                # ------------------------------------------------

                update(
                    0.02,
                    "📥 Video ကို ပြင်ဆင်နေသည်...",
                )

                video_path.write_bytes(
                    uploaded_video.getbuffer()
                )

                total_duration = (
                    get_duration(
                        video_path
                    )
                )

                # ------------------------------------------------
                # AUDIO
                # ------------------------------------------------

                update(
                    0.08,
                    "🎵 Audio ထုတ်နေသည်...",
                )

                extract_audio(
                    video_path,
                    audio_path,
                )

                # ------------------------------------------------
                # GEMINI
                # ------------------------------------------------

                update(
                    0.15,
                    "🤖 Gemini ကို Audio ပို့နေသည်...",
                )

                client = genai.Client(
                    api_key=api_key
                )

                uploaded_audio = (
                    upload_audio(
                        client,
                        audio_path,
                    )
                )

                # ------------------------------------------------
                # AI ANALYSIS
                # ------------------------------------------------

                update(
                    0.20,
                    "🧠 Dialogue တွေကို AI က စစ်ဆေးနေသည်...",
                )

                model_name, raw_segments = (
                    analyze_audio(
                        client,
                        uploaded_audio,
                        status_callback=lambda msg: status.write(
                            f"**{msg}**"
                        ),
                    )
                )

                segments = clean_segments(
                    raw_segments,
                    total_duration,
                )

                if not segments:
                    raise RuntimeError(
                        "AI မှ Dialogue မတွေ့ပါ။"
                    )

                st.success(
                    f"✅ Dialogue {len(segments)} ခု "
                    f"တွေ့ပါပြီ • Model: "
                    f"{model_name}"
                )

                # ------------------------------------------------
                # SCRIPT
                # ------------------------------------------------

                with st.expander(
                    "📝 Burmese Script",
                    expanded=False,
                ):

                    for index, segment in enumerate(
                        segments,
                        start=1,
                    ):

                        st.write(
                            f"**{index}.** "
                            f"{segment['start']:.2f}s → "
                            f"{segment['end']:.2f}s"
                        )

                        st.write(
                            segment["text"]
                        )

                # ------------------------------------------------
                # TTS + SYNC
                # ------------------------------------------------

                update(
                    0.40,
                    "🎙️ Burmese Voice ဖန်တီးနေသည်...",
                )

                final_audio = (
                    build_burmese_audio(
                        segments,
                        VOICE_MAP[
                            voice_name
                        ],
                        total_duration,
                        work_dir,
                        progress_callback=update,
                    )
                )

                # ------------------------------------------------
                # FINAL VIDEO
                # ------------------------------------------------

                update(
                    0.80,
                    "🎬 Final Video တည်ဆောက်နေသည်...",
                )

                merge_video_audio(
                    video_path,
                    final_audio,
                    output_path,
                )

                update(
                    0.96,
                    "🔍 Final Video စစ်ဆေးနေသည်...",
                )

                if (
                    not output_path.exists()
                    or
                    output_path.stat().st_size
                    < 10000
                ):
                    raise RuntimeError(
                        "Final video file မထွက်ပါ။"
                    )

                final_bytes = (
                    output_path.read_bytes()
                )

                update(
                    1.0,
                    "✅ အားလုံးပြီးပါပြီ!",
                )

                st.success(
                    "🎉 Myanmar Dubbed Video Ready!"
                )

                st.video(
                    final_bytes
                )

                st.download_button(
                    label="⬇️ DOWNLOAD FINAL VIDEO",
                    data=final_bytes,
                    file_name=(
                        Path(
                            uploaded_video.name
                        ).stem
                        +
                        "_Myanmar_Dub.mp4"
                    ),
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True,
                )

        except Exception as error:

            st.error(
                "❌ Processing Error"
            )

            st.code(
                str(error),
                language="text",
            )
