import os
import re
import json
import time
import random
import asyncio
import subprocess
import tempfile
from pathlib import Path

import requests
import streamlit as st
import edge_tts
import imageio_ffmpeg
from google import genai


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Myanmar Movie AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Gemini model fallback list
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

VOICES = {
    "သီဟ (အမျိုးသား)": "my-MM-ThihaNeural",
    "နီလာ (အမျိုးသမီး)": "my-MM-NilarNeural",
}

VOICE_STYLES = {
    "ပုံမှန်": {
        "rate": 0,
        "pitch": 0,
    },
    "နက်နက် (Deep)": {
        "rate": -5,
        "pitch": -12,
    },
    "ပျော့ပျောင်း": {
        "rate": -3,
        "pitch": 5,
    },
    "တက်ကြွ": {
        "rate": 8,
        "pitch": 2,
    },
}

RETRY_STATUS = {
    429,
    500,
    502,
    503,
    504,
}


# ============================================================
# UI STYLE
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at top right,
            rgba(90, 60, 160, 0.18),
            transparent 35%
        ),
        radial-gradient(
            circle at top left,
            rgba(30, 100, 180, 0.12),
            transparent 30%
        ),
        #090b12;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 28px;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(40, 45, 70, 0.95),
            rgba(18, 20, 32, 0.98)
        );
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 24px;
    box-shadow: 0 15px 50px rgba(0,0,0,0.25);
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 6px;
}

.hero-subtitle {
    color: #aeb4c5;
    font-size: 15px;
}

.step-card {
    padding: 22px;
    border-radius: 20px;
    background: rgba(20,23,34,0.85);
    border: 1px solid rgba(255,255,255,0.07);
    margin: 12px 0 20px 0;
}

.status-pill {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(80,180,120,0.12);
    color: #8de0aa;
    font-size: 12px;
    margin-right: 6px;
}

div[data-testid="stFileUploader"] {
    border-radius: 16px;
}

button[kind="primary"] {
    border-radius: 14px !important;
    font-weight: 700 !important;
}

.download-row {
    margin-top: 10px;
}

.small-muted {
    color: #8e95a7;
    font-size: 13px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def get_secret(name):
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""

    if value:
        return str(value).strip()

    return os.getenv(name, "").strip()


def safe_filename(name):
    name = str(name or "").strip()

    name = re.sub(
        r'[\\/:*?"<>|]+',
        "_",
        name,
    )

    name = re.sub(
        r"\s+",
        "_",
        name,
    )

    name = name.strip("._")

    return name or "output"


def clean_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text or ""),
    ).strip()


def run_cmd(args, timeout=1800):
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
    )


def ffprobe_duration(path):
    result = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(path),
        ],
        120,
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr or "",
    )

    if not match:
        raise RuntimeError(
            "Media duration ကို ဖတ်မရပါ။"
        )

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )


# ============================================================
# FFMPEG
# ============================================================

def extract_audio(video_path, audio_path):
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
            str(audio_path),
        ],
        900,
    )

    if (
        result.returncode != 0
        or not audio_path.exists()
        or audio_path.stat().st_size < 1000
    ):
        raise RuntimeError(
            "Video ထဲက audio ထုတ်မရပါ။\n"
            + (result.stderr or "")
        )


# ============================================================
# GEMINI
# ============================================================

def get_gemini_client():
    key = get_secret("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ကို စစ်ပါ။"
        )

    return genai.Client(
        api_key=key
    )


def clean_json_response(text):
    text = str(text or "").strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
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

    return text


def should_retry_error(error):
    text = str(error).lower()

    words = [
        "503",
        "429",
        "500",
        "502",
        "504",
        "timeout",
        "unavailable",
        "overloaded",
        "resource exhausted",
        "high demand",
    ]

    return any(
        word in text
        for word in words
    )


def translate_batch(client, items):
    prompt = f"""
You are a professional movie subtitle translator.

Translate the following movie dialogue into natural,
conversational Burmese used by Myanmar viewers.

IMPORTANT RULES:

1. Preserve the original meaning.
2. Do not summarize.
3. Do not explain anything.
4. Preserve names and proper nouns.
5. Preserve emotion and speaking tone.
6. Make Burmese sound natural when spoken aloud.
7. Keep each translation reasonably short for its timestamp.
8. Do not add quotation marks.
9. Do not add timestamps.
10. Return ONLY JSON.
11. Return exactly the same number of items.
12. Keep the same id.

JSON FORMAT:

[
  {{"id": 1, "burmese": "..."}}
]

SOURCE:

{json.dumps(items, ensure_ascii=False)}
"""

    errors = []

    for model in GEMINI_MODELS:

        for attempt in range(2):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                raw = getattr(
                    response,
                    "text",
                    "",
                )

                data = json.loads(
                    clean_json_response(raw)
                )

                if not isinstance(
                    data,
                    list,
                ):
                    raise RuntimeError(
                        "Gemini က JSON array မပြန်ပါ။"
                    )

                if len(data) != len(items):
                    raise RuntimeError(
                        "Translation result count မကိုက်ပါ။"
                    )

                translated = {}

                for item in data:

                    idx = int(
                        item["id"]
                    )

                    burmese = clean_text(
                        item.get(
                            "burmese",
                            "",
                        )
                    )

                    if burmese:
                        translated[idx] = burmese

                for i in range(
                    1,
                    len(items) + 1,
                ):
                    if i not in translated:
                        raise RuntimeError(
                            f"Translation #{i} မထွက်ပါ။"
                        )

                return translated

            except Exception as error:

                errors.append(
                    f"{model} / attempt {attempt + 1}: "
                    f"{error}"
                )

                if (
                    attempt == 0
                    and should_retry_error(error)
                ):
                    time.sleep(
                        2 + random.random() * 2
                    )
                else:
                    break

    raise RuntimeError(
        "Gemini ဘာသာပြန်ခြင်း မအောင်မြင်ပါ။\n\n"
        + "\n".join(
            errors[-8:]
        )
    )


# ============================================================
# DEEPGRAM
# ============================================================

def get_deepgram_key():
    key = get_secret(
        "DEEPGRAM_API_KEY"
    )

    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ကို စစ်ပါ။"
        )

    return key


def words_to_segments(words):
    MAX_CHARS = 42
    MAX_DURATION = 6.0
    PAUSE_SPLIT = 0.65

    segments = []
    current = []

    def flush():

        nonlocal current

        if not current:
            return

        first = current[0]
        last = current[-1]

        words_text = []

        for word in current:

            value = word.get(
                "punctuated_word"
            )

            if not value:
                value = word.get(
                    "word",
                    "",
                )

            if value:
                words_text.append(
                    str(value)
                )

        text = clean_text(
            " ".join(words_text)
        )

        start = float(
            first.get(
                "start",
                0,
            )
        )

        end = float(
            last.get(
                "end",
                start,
            )
        )

        if (
            text
            and end > start
        ):
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "source": text,
                }
            )

        current = []

    for word in words:

        if not word.get("word"):
            continue

        if current:

            previous = current[-1]

            previous_end = float(
                previous.get(
                    "end",
                    previous.get(
                        "start",
                        0,
                    ),
                )
            )

            current_start = float(
                word.get(
                    "start",
                    previous_end,
                )
            )

            pause = (
                current_start
                - previous_end
            )

            current_text = " ".join(
                str(
                    x.get(
                        "punctuated_word",
                        x.get(
                            "word",
                            "",
                        ),
                    )
                )
                for x in current
            )

            new_word = str(
                word.get(
                    "punctuated_word",
                    word.get(
                        "word",
                        "",
                    ),
                )
            )

            proposed_text = (
                current_text
                + " "
                + new_word
            )

            first_start = float(
                current[0].get(
                    "start",
                    current_start,
                )
            )

            word_end = float(
                word.get(
                    "end",
                    current_start,
                )
            )

            proposed_duration = (
                word_end
                - first_start
            )

            if (
                pause >= PAUSE_SPLIT
                or len(proposed_text) > MAX_CHARS
                or proposed_duration > MAX_DURATION
            ):
                flush()

        current.append(word)

        punctuation = str(
            word.get(
                "punctuated_word",
                word.get(
                    "word",
                    "",
                ),
            )
        )

        if punctuation.endswith(
            (
                ".",
                "!",
                "?",
                "。",
                "！",
                "？",
            )
        ):
            flush()

    flush()

    return segments


def deepgram_transcribe(audio_path):

    url = (
        "https://api.deepgram.com/v1/listen"
    )

    params = {
        "model": "nova-3",
        "detect_language": "true",
        "punctuate": "true",
        "smart_format": "true",
        "utterances": "true",
        "diarize": "true",
        "words": "true",
    }

    headers = {
        "Authorization": (
            f"Token {get_deepgram_key()}"
        ),
        "Content-Type": "audio/wav",
    }

    audio_data = audio_path.read_bytes()

    last_error = ""

    for attempt in range(3):

        try:

            response = requests.post(
                url,
                params=params,
                headers=headers,
                data=audio_data,
                timeout=900,
            )

            if response.status_code == 200:

                data = response.json()

                results = data.get(
                    "results",
                    {},
                )

                channels = results.get(
                    "channels",
                    [],
                )

                words = []

                if channels:

                    alternatives = (
                        channels[0].get(
                            "alternatives",
                            [],
                        )
                    )

                    if alternatives:

                        words = (
                            alternatives[0].get(
                                "words",
                                [],
                            )
                            or []
                        )

                if words:

                    segments = (
                        words_to_segments(
                            words
                        )
                    )

                    if segments:
                        return segments

                # Fallback to utterances
                utterances = (
                    results.get(
                        "utterances",
                        [],
                    )
                    or []
                )

                fallback = []

                for utterance in utterances:

                    text = clean_text(
                        utterance.get(
                            "transcript",
                            "",
                        )
                    )

                    start = float(
                        utterance.get(
                            "start",
                            0,
                        )
                    )

                    end = float(
                        utterance.get(
                            "end",
                            start,
                        )
                    )

                    if (
                        text
                        and end > start
                    ):
                        fallback.append(
                            {
                                "start": start,
                                "end": end,
                                "source": text,
                            }
                        )

                if fallback:
                    return fallback

                raise RuntimeError(
                    "Deepgram transcript မတွေ့ပါ။"
                )

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

            if response.status_code not in RETRY_STATUS:
                break

        except Exception as error:
            last_error = str(error)

        time.sleep(
            (2 ** attempt)
            + random.random()
        )

    raise RuntimeError(
        "Deepgram STT မအောင်မြင်ပါ။\n"
        + last_error
    )


# ============================================================
# SRT
# ============================================================

def seconds_to_srt(seconds):

    seconds = max(
        0.0,
        float(seconds),
    )

    total_ms = int(
        round(seconds * 1000)
    )

    hours, remainder = divmod(
        total_ms,
        3600000,
    )

    minutes, remainder = divmod(
        remainder,
        60000,
    )

    secs, milliseconds = divmod(
        remainder,
        1000,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def make_srt(segments):

    blocks = []

    for index, item in enumerate(
        segments,
        1,
    ):

        blocks.append(
            f"{index}\n"
            f"{seconds_to_srt(item['start'])}"
            f" --> "
            f"{seconds_to_srt(item['end'])}\n"
            f"{item['burmese']}\n"
        )

    return "\n".join(blocks)


def parse_srt_time(value):

    value = (
        value
        .strip()
        .replace(",", ".")
    )

    parts = value.split(":")

    if len(parts) != 3:
        raise ValueError(
            "Invalid SRT timestamp"
        )

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )


def parse_srt(text):

    text = (
        str(text or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\ufeff", "")
        .strip()
    )

    if not text:
        raise RuntimeError(
            "SRT ဖိုင်အလွတ်ဖြစ်နေပါသည်။"
        )

    blocks = re.split(
        r"\n\s*\n",
        text,
    )

    parsed = []

    for block in blocks:

        lines = [
            line.strip()
            for line in block.split("\n")
        ]

        lines = [
            line
            for line in lines
            if line
        ]

        if len(lines) < 2:
            continue

        timestamp_index = -1

        for i, line in enumerate(lines):

            if "-->" in line:
                timestamp_index = i
                break

        if timestamp_index < 0:
            continue

        timestamp = lines[
            timestamp_index
        ]

        match = re.match(
            r"\s*"
            r"(\d{2}:\d{2}:\d{2}[,.]\d{1,3})"
            r"\s*-->\s*"
            r"(\d{2}:\d{2}:\d{2}[,.]\d{1,3})",
            timestamp,
        )

        if not match:
            continue

        start = parse_srt_time(
            match.group(1)
        )

        end = parse_srt_time(
            match.group(2)
        )

        subtitle_lines = lines[
            timestamp_index + 1:
        ]

        subtitle = clean_text(
            " ".join(
                subtitle_lines
            )
        )

        if not subtitle:
            continue

        parsed.append(
            {
                "start": start,
                "end": end,
                "burmese": subtitle,
            }
        )

    if not parsed:
        raise RuntimeError(
            "Valid subtitle မတွေ့ပါ။"
        )

    parsed.sort(
        key=lambda x: x["start"]
    )

    cleaned = []
    fixed_count = 0

    for item in parsed:

        start = max(
            0.0,
            float(item["start"]),
        )

        end = max(
            0.0,
            float(item["end"]),
        )

        if end <= start:
            fixed_count += 1
            continue

        if cleaned:

            previous_end = (
                cleaned[-1]["end"]
            )

            if start < previous_end:

                start = previous_end
                fixed_count += 1

        if end <= start:
            fixed_count += 1
            continue

        cleaned.append(
            {
                "start": start,
                "end": end,
                "burmese": clean_text(
                    item["burmese"]
                ),
            }
        )

    if not cleaned:
        raise RuntimeError(
            "SRT timing မှာ valid subtitle မကျန်ပါ။"
        )

    return cleaned, fixed_count


# ============================================================
# BUILD SRT
# ============================================================

def translate_segments(
    source_segments,
    client,
    progress_callback,
):

    valid = []

    for item in source_segments:

        text = clean_text(
            item.get(
                "source",
                "",
            )
        )

        start = float(
            item.get(
                "start",
                0,
            )
        )

        end = float(
            item.get(
                "end",
                start,
            )
        )

        if (
            text
            and end > start
        ):
            valid.append(
                {
                    "start": start,
                    "end": end,
                    "source": text,
                }
            )

    if not valid:
        raise RuntimeError(
            "Dialogue မတွေ့ပါ။"
        )

    translated_segments = []

    batch_size = 10

    total = len(valid)

    for start_index in range(
        0,
        total,
        batch_size,
    ):

        batch = valid[
            start_index:
            start_index + batch_size
        ]

        payload = []

        for i, item in enumerate(
            batch,
            1,
        ):
            payload.append(
                {
                    "id": i,
                    "text": item["source"],
                }
            )

        progress_callback(
            0.30
            + (
                0.55
                * start_index
                / max(total, 1)
            ),
            (
                "Gemini ဘာသာပြန်နေသည်... "
                f"{min(start_index + len(batch), total)}"
                f"/{total}"
            ),
        )

        result = translate_batch(
            client,
            payload,
        )

        for i, item in enumerate(
            batch,
            1,
        ):

            translated_segments.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "burmese": result[i],
                }
            )

    return translated_segments


# ============================================================
# EDGE TTS
# ============================================================

async def save_edge_tts(
    text,
    voice,
    rate,
    pitch,
    output,
):

    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=f"{rate:+d}%",
        pitch=f"{pitch:+d}Hz",
    )

    await communicator.save(
        str(output)
    )


def generate_tts(
    text,
    voice,
    style,
    output,
):

    settings = VOICE_STYLES[
        style
    ]

    errors = []

    for attempt in range(3):

        try:

            if output.exists():
                output.unlink()

            asyncio.run(
                save_edge_tts(
                    text,
                    voice,
                    settings["rate"],
                    settings["pitch"],
                    output,
                )
            )

            if (
                output.exists()
                and output.stat().st_size > 1000
            ):
                return

            raise RuntimeError(
                "TTS output အလွတ်ဖြစ်နေပါသည်။"
            )

        except Exception as error:

            errors.append(
                str(error)
            )

            time.sleep(
                1.5 + attempt
            )

    raise RuntimeError(
        "Burmese TTS မအောင်မြင်ပါ။\n"
        + "\n".join(
            errors[-3:]
        )
    )


# ============================================================
# AUDIO SPEED / FIT
# ============================================================

def build_atempo(speed):

    speed = max(
        0.25,
        min(
            float(speed),
            4.0,
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

    return ",".join(filters)


def fit_audio_to_slot(
    source,
    output,
    slot_duration,
    user_speed,
):

    raw_duration = ffprobe_duration(
        source
    )

    slot_duration = max(
        0.05,
        float(slot_duration),
    )

    user_speed = max(
        0.70,
        min(
            float(user_speed),
            1.30,
        ),
    )

    # Automatically calculate speed needed
    # to fit the speech into the subtitle slot.
    fit_speed = (
        raw_duration
        / slot_duration
    )

    final_speed = (
        fit_speed
        * user_speed
    )

    final_speed = max(
        0.25,
        min(
            final_speed,
            4.0,
        ),
    )

    audio_filter = (
        build_atempo(final_speed)
        + ",apad,"
        + f"atrim=duration={slot_duration:.3f}"
    )

    result = run_cmd(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-filter:a",
            audio_filter,
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(output),
        ],
        180,
    )

    if (
        result.returncode != 0
        or not output.exists()
        or output.stat().st_size < 1000
    ):
        raise RuntimeError(
            "Voice timing ပြင်မရပါ။\n"
            + (result.stderr or "")
        )


# ============================================================
# BUILD FINAL VOICEOVER
# ============================================================

def build_voiceover(
    segments,
    voice,
    style,
    speed,
    work_dir,
    progress_callback,
):

    clips = []

    total = len(segments)

    for index, item in enumerate(
        segments,
        1,
    ):

        start = max(
            0.0,
            float(item["start"]),
        )

        end = max(
            start + 0.05,
            float(item["end"]),
        )

        slot = end - start

        raw_file = (
            work_dir
            / f"tts_{index:04d}.mp3"
        )

        fitted_file = (
            work_dir
            / f"voice_{index:04d}.m4a"
        )

        progress_callback(
            0.08
            + 0.70
            * ((index - 1) / max(total, 1)),
            (
                "Voiceover ထုတ်နေသည်... "
                f"{index}/{total}"
            ),
        )

        generate_tts(
            item["burmese"],
            voice,
            style,
            raw_file,
        )

        fit_audio_to_slot(
            raw_file,
            fitted_file,
            slot,
            speed,
        )

        clips.append(
            (
                start,
                fitted_file,
            )
        )

    if not clips:
        raise RuntimeError(
            "Voiceover ပြုလုပ်ရန် subtitle မရှိပါ။"
        )

    output = (
        work_dir
        / "burmese_voiceover.m4a"
    )

    command = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    for _, file_path in clips:

        command.extend(
            [
                "-i",
                str(file_path),
            ]
        )

    filters = []
    labels = []

    for index, (
        start,
        _,
    ) in enumerate(clips):

        delay_ms = max(
            0,
            int(
                round(
                    start * 1000
                )
            ),
        )

        label = f"a{index}"

        filters.append(
            f"[{index}:a]"
            f"adelay={delay_ms}:all=1,"
            "aresample=48000"
            f"[{label}]"
        )

        labels.append(
            f"[{label}]"
        )

    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:"
        "duration=longest:"
        "dropout_transition=0,"
        "volume=1.5,"
        "loudnorm="
        "I=-16:"
        "TP=-1.5:"
        "LRA=11,"
        "alimiter=limit=0.95"
        "[out]"
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )

    result = run_cmd(
        command,
        1800,
    )

    if (
        result.returncode != 0
        or not output.exists()
        or output.stat().st_size < 5000
    ):
        raise RuntimeError(
            "Final Voiceover မထုတ်နိုင်ပါ။\n"
            + (result.stderr or "")
        )

    # Validate output
    check = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(output),
            "-f",
            "null",
            "-",
        ],
        300,
    )

    if check.returncode != 0:
        raise RuntimeError(
            "Voiceover validation မအောင်မြင်ပါ။\n"
            + (check.stderr or "")
        )

    progress_callback(
        1.0,
        "Voiceover ပြီးပါပြီ။",
    )

    return output


# ============================================================
# SESSION STATE
# ============================================================

if "srt_text" not in st.session_state:
    st.session_state.srt_text = ""

if "srt_name" not in st.session_state:
    st.session_state.srt_name = "myanmar.srt"

if "voice_bytes" not in st.session_state:
    st.session_state.voice_bytes = None

if "voice_filename" not in st.session_state:
    st.session_state.voice_filename = (
        "myanmar_voiceover.m4a"
    )

if "voice_segments" not in st.session_state:
    st.session_state.voice_segments = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="hero-title">
🎬 Myanmar Movie AI
</div>

<div class="hero-subtitle">
Movie dialogue ကို မြန်မာ SRT အဖြစ်ပြောင်းပြီး
သဘာဝကျတဲ့ မြန်မာ Voiceover ထုတ်ပေးမယ့် AI Studio
</div>

<br>

<span class="status-pill">Deepgram STT</span>
<span class="status-pill">Gemini AI</span>
<span class="status-pill">Myanmar TTS</span>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# STEP 1
# ============================================================

st.markdown(
    """
<div class="step-card">

<h2>① Video → မြန်မာ SRT</h2>

<div class="small-muted">
Video ထဲက dialogue ကို timestamp အတိအကျဖတ်ပြီး
Gemini နဲ့ သဘာဝကျတဲ့ မြန်မာစာအဖြစ် ဘာသာပြန်ပေးမယ်။
</div>

</div>
""",
    unsafe_allow_html=True,
)

video_file = st.file_uploader(
    "🎥 Movie Video တင်ပါ",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm",
    ],
    key="main_video",
)


with st.form(
    "srt_form",
    clear_on_submit=False,
):

    create_srt = st.form_submit_button(
        "📝  မြန်မာ SRT ထုတ်မယ်",
        type="primary",
        use_container_width=True,
    )


if create_srt:

    if not video_file:

        st.error(
            "အရင်ဆုံး Video တစ်ခု တင်ပါ။"
        )

        st.stop()

    try:

        with tempfile.TemporaryDirectory() as temp:

            work = Path(temp)

            video_path = (
                work / "input_video"
            )

            audio_path = (
                work / "audio.wav"
            )

            video_path.write_bytes(
                video_file.getvalue()
            )

            status = st.empty()

            progress_bar = st.progress(
                0.0
            )

            status.info(
                "🎧 Video audio ထုတ်နေသည်..."
            )

            extract_audio(
                video_path,
                audio_path,
            )

            progress_bar.progress(
                0.15
            )

            status.info(
                "🎙️ Deepgram က dialogue "
                "နဲ့ timestamp ရယူနေသည်..."
            )

            source_segments = (
                deepgram_transcribe(
                    audio_path
                )
            )

            progress_bar.progress(
                0.30
            )

            status.info(
                f"တွေ့ရှိသော dialogue — "
                f"{len(source_segments)} lines"
            )

            client = (
                get_gemini_client()
            )

            translated = (
                translate_segments(
                    source_segments,
                    client,
                    lambda value, message: (
                        progress_bar.progress(
                            min(
                                max(
                                    float(value),
                                    0.0,
                                ),
                                1.0,
                            )
                        ),
                        status.info(
                            message
                        ),
                    ),
                )
            )

            srt_text = make_srt(
                translated
            )

            st.session_state.srt_text = (
                srt_text
            )

            st.session_state.srt_name = (
                safe_filename(
                    Path(
                        video_file.name
                    ).stem
                    + "_Myanmar"
                )
                + ".srt"
            )

            st.session_state.voice_segments = (
                translated
            )

            progress_bar.progress(
                1.0
            )

            status.success(
                f"✅ Myanmar SRT ပြီးပါပြီ — "
                f"{len(translated)} lines"
            )

    except Exception as error:

        st.error(
            "❌ SRT ထုတ်ရာမှာ အမှားဖြစ်ပါတယ်။"
        )

        st.exception(error)


# ============================================================
# SRT RESULT
# ============================================================

if st.session_state.srt_text:

    st.markdown(
        "### 📄 Myanmar SRT Preview"
    )

    st.text_area(
        "SRT",
        value=st.session_state.srt_text,
        height=320,
        key="srt_preview",
    )

    st.download_button(
        "⬇️ Download Myanmar SRT",
        data=st.session_state.srt_text.encode(
            "utf-8-sig"
        ),
        file_name=st.session_state.srt_name,
        mime="application/x-subrip",
        use_container_width=True,
    )


# ============================================================
# STEP 2
# ============================================================

st.markdown(
    """
<div class="step-card">

<h2>② SRT → မြန်မာ Voiceover</h2>

<div class="small-muted">
SRT timestamp ကို အလိုအလျောက်စစ်ပြီး
မြန်မာ Voice ကို timestamp နဲ့ကိုက်အောင် ထုတ်ပေးမယ်။
</div>

</div>
""",
    unsafe_allow_html=True,
)

srt_file = st.file_uploader(
    "📄 SRT ဖိုင်တင်ပါ",
    type=["srt"],
    key="voice_srt",
)


with st.form(
    "voice_form",
    clear_on_submit=False,
):

    voice_name = st.selectbox(
        "🎙️ Voice",
        options=list(VOICES.keys()),
    )

    style = st.selectbox(
        "🎭 Voice Style",
        options=list(
            VOICE_STYLES.keys()
        ),
    )

    speed = st.slider(
        "⚡ Speed",
        min_value=0.70,
        max_value=1.30,
        value=1.00,
        step=0.05,
        help=(
            "SRT timing အတွင်း အသံကို "
            "ပိုမြန်/ပိုနှေးအောင် ချိန်ပေးသည်။"
        ),
    )

    output_filename = st.text_input(
        "💾 Voiceover Filename",
        value="myanmar_voiceover",
    )

    create_voice = st.form_submit_button(
        "🗣️  Voiceover ထုတ်မယ်",
        type="primary",
        use_container_width=True,
    )


if create_voice:

    source_srt = ""

    # Uploaded SRT gets priority
    if srt_file:

        source_srt = (
            srt_file.getvalue()
            .decode(
                "utf-8-sig",
                errors="replace",
            )
        )

    elif st.session_state.srt_text:

        source_srt = (
            st.session_state.srt_text
        )

    else:

        st.error(
            "SRT ဖိုင်တင်ပါ "
            "(သို့မဟုတ် Step 1 မှာ SRT အရင်ထုတ်ပါ)။"
        )

        st.stop()

    try:

        segments, fixed_count = (
            parse_srt(
                source_srt
            )
        )

        with tempfile.TemporaryDirectory() as temp:

            work = Path(temp)

            status = st.empty()

            progress_bar = st.progress(
                0.0
            )

            if fixed_count:

                status.warning(
                    f"⚠️ SRT timing ကို "
                    f"{fixed_count} နေရာ "
                    f"အလိုအလျောက်ပြင်ထားပါတယ်။"
                )

            else:

                status.info(
                    f"✅ SRT timing OK — "
                    f"{len(segments)} lines"
                )

            voice_path = (
                build_voiceover(
                    segments,
                    VOICES[voice_name],
                    style,
                    speed,
                    work,
                    lambda value, message: (
                        progress_bar.progress(
                            min(
                                max(
                                    float(value),
                                    0.0,
                                ),
                                1.0,
                            )
                        ),
                        status.info(
                            message
                        ),
                    ),
                )
            )

            voice_bytes = (
                voice_path.read_bytes()
            )

            filename = safe_filename(
                output_filename
            )

            if not filename.lower().endswith(
                ".m4a"
            ):
                filename += ".m4a"

            st.session_state.voice_bytes = (
                voice_bytes
            )

            st.session_state.voice_filename = (
                filename
            )

            st.session_state.voice_segments = (
                segments
            )

            progress_bar.progress(
                1.0
            )

            status.success(
                "✅ Voiceover အောင်မြင်ပါပြီ။"
            )

    except Exception as error:

        st.error(
            "❌ Voiceover ထုတ်ရာမှာ "
            "အမှားဖြစ်ပါတယ်။"
        )

        st.exception(error)


# ============================================================
# VOICE RESULT
# ============================================================

if st.session_state.voice_bytes:

    st.markdown(
        "### 🔊 Voiceover Preview"
    )

    st.audio(
        st.session_state.voice_bytes,
        format="audio/mp4",
    )

    st.download_button(
        "⬇️ Download Voiceover",
        data=st.session_state.voice_bytes,
        file_name=st.session_state.voice_filename,
        mime="audio/mp4",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🎬 Myanmar Movie AI • "
    "Video → Myanmar SRT → Burmese Voiceover"
)
