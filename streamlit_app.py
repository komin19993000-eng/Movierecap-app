import os
import re
import json
import time
import random
import subprocess
import tempfile
from pathlib import Path

import requests
import streamlit as st
import edge_tts
import asyncio
import imageio_ffmpeg
from google import genai


st.set_page_config(
    page_title="Myanmar Movie AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# =========================================================
# UI
# =========================================================

st.markdown(
"""<style>

/* ================================
   MAIN BACKGROUND
================================ */

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(0, 102, 255, 0.30),
            transparent 35%
        ),
        radial-gradient(
            circle at 100% 20%,
            rgba(255, 0, 55, 0.28),
            transparent 35%
        ),
        linear-gradient(
            145deg,
            #050b18 0%,
            #08142b 45%,
            #12060d 100%
        );
}

[data-testid="stHeader"] {
    background: transparent;
}

/* ================================
   MAIN CONTENT
================================ */

.block-container {
    max-width: 1050px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* ================================
   ALL NORMAL TEXT
================================ */

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div {
    color: #ffffff;
}

/* ================================
   HERO
================================ */

.hero {
    padding: 30px 24px;
    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            rgba(0, 100, 255, 0.28),
            rgba(255, 0, 65, 0.25)
        );

    border: 1px solid rgba(255,255,255,0.20);

    box-shadow:
        0 15px 50px rgba(0,0,0,0.55),
        0 0 35px rgba(0,100,255,0.12);

    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    margin: 0 0 10px 0 !important;

    text-shadow:
        0 0 12px rgba(0,140,255,0.65);
}

.hero p {
    color: #f5f7ff !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}

/* ================================
   BADGES
================================ */

.badge {
    display: inline-block;

    padding: 7px 13px;
    margin: 4px;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            rgba(0,120,255,0.75),
            rgba(255,0,70,0.75)
        );

    color: #ffffff !important;

    font-size: 0.88rem !important;
    font-weight: 800 !important;

    border: 1px solid rgba(255,255,255,0.25);

    box-shadow:
        0 4px 15px rgba(0,0,0,0.30);
}

/* ================================
   STEP TITLES
================================ */

.step {
    font-size: 1.55rem !important;
    font-weight: 900 !important;

    color: #ffffff !important;

    margin-top: 28px;
    margin-bottom: 10px;

    text-shadow:
        0 0 12px rgba(0,120,255,0.55);
}

/* ================================
   CAPTION
================================ */

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #dce6ff !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

/* ================================
   CARDS
================================ */

.card {
    padding: 20px;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(20,40,75,0.92),
            rgba(55,15,25,0.90)
        );

    border:
        1px solid rgba(255,255,255,0.14);

    box-shadow:
        0 10px 35px rgba(0,0,0,0.45);
}

/* ================================
   HEADINGS
================================ */

h1,
h2,
h3,
.stSubheader {
    color: #ffffff !important;
    font-weight: 900 !important;
}

/* ================================
   FILE UPLOADER
================================ */

[data-testid="stFileUploader"] {
    background:
        linear-gradient(
            135deg,
            rgba(0,95,255,0.18),
            rgba(255,0,60,0.15)
        );

    border:
        2px dashed rgba(90,160,255,0.65);

    border-radius: 18px;

    padding: 8px;

    box-shadow:
        0 5px 25px rgba(0,0,0,0.30);
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
}

[data-testid="stFileUploader"] button {
    background:
        linear-gradient(
            90deg,
            #087cff,
            #ff174d
        ) !important;

    color: #ffffff !important;

    border: none !important;

    font-weight: 900 !important;

    border-radius: 12px !important;
}

/* ================================
   SELECT BOX
================================ */

[data-baseweb="select"] > div {
    background: #f7f9ff !important;

    border:
        2px solid rgba(40,120,255,0.55) !important;

    border-radius: 14px !important;

    min-height: 50px !important;
}

[data-baseweb="select"] span {
    color: #101827 !important;
    font-weight: 800 !important;
}

[data-baseweb="select"] svg {
    fill: #176cff !important;
}

/* ================================
   SLIDER
================================ */

[data-testid="stSlider"] label {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
}

/* ================================
   BUTTONS
================================ */

div.stButton > button {

    min-height: 52px !important;

    border-radius: 15px !important;

    border: none !important;

    background:
        linear-gradient(
            90deg,
            #006eff 0%,
            #174cff 48%,
            #ff174d 100%
        ) !important;

    color: #ffffff !important;

    font-size: 1.05rem !important;

    font-weight: 900 !important;

    box-shadow:
        0 8px 25px rgba(0,80,255,0.30);

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease;
}

div.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 10px 30px rgba(255,30,80,0.35);
}

/* ================================
   DOWNLOAD BUTTON
================================ */

[data-testid="stDownloadButton"] button {

    min-height: 50px !important;

    border-radius: 14px !important;

    background:
        linear-gradient(
            90deg,
            #008cff,
            #005eff
        ) !important;

    color: #ffffff !important;

    font-size: 1rem !important;

    font-weight: 900 !important;

    border: none !important;

    box-shadow:
        0 7px 20px rgba(0,100,255,0.30);
}

/* ================================
   TEXT AREA / SRT PREVIEW
================================ */

[data-testid="stTextArea"] textarea {

    background:
        #f7f9ff !important;

    color:
        #111827 !important;

    border:
        2px solid rgba(50,120,255,0.45) !important;

    border-radius:
        16px !important;

    font-size:
        15px !important;

    font-weight:
        600 !important;

    line-height:
        1.7 !important;
}

/* ================================
   INFO BOX
================================ */

[data-testid="stAlert"] {

    border-radius: 15px !important;

    border:
        1px solid rgba(90,160,255,0.35) !important;

    background:
        rgba(10,55,120,0.55) !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ================================
   METRICS
================================ */

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.8rem !important;
    font-weight: 900 !important;
}

[data-testid="stMetricLabel"] {
    color: #c9d7ff !important;
    font-weight: 700 !important;
}

/* ================================
   MOBILE
================================ */

@media (max-width: 600px) {

    .block-container {
        padding-left: 14px;
        padding-right: 14px;
        padding-top: 1rem;
    }

    .hero {
        padding: 23px 17px;
        border-radius: 20px;
    }

    .hero h1 {
        font-size: 2rem !important;
    }

    .hero p {
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }

    .step {
        font-size: 1.3rem !important;
    }

    .badge {
        font-size: 0.78rem !important;
        padding: 6px 9px;
    }

    div.stButton > button {
        min-height: 54px !important;
        font-size: 1rem !important;
    }
}

</style>""",
    unsafe_allow_html=True,
)


# =========================================================
# CONFIG
# =========================================================

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

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

RETRY_WORDS = (
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
)


# =========================================================
# COMMON
# =========================================================

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
    r = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(path),
        ],
        120,
    )

    m = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        r.stderr or "",
    )

    if not m:
        raise RuntimeError("Video/audio duration ကို ဖတ်မရပါ။")

    return (
        int(m.group(1)) * 3600
        + int(m.group(2)) * 60
        + float(m.group(3))
    )


def get_secret(name):
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass

    return os.getenv(name, "").strip()


# =========================================================
# AUDIO EXTRACTION
# =========================================================

def extract_audio(video, out):
    r = run_cmd(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out),
        ],
        900,
    )

    if (
        r.returncode != 0
        or not out.exists()
        or out.stat().st_size < 1000
    ):
        raise RuntimeError(
            "Video ထဲက audio ထုတ်မရပါ။\n"
            + (r.stderr or "")
        )


# =========================================================
# GEMINI
# =========================================================

def gemini_client():
    key = get_secret("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return genai.Client(api_key=key)


# =========================================================
# DEEPGRAM
# =========================================================

def deepgram_key():
    key = get_secret("DEEPGRAM_API_KEY")

    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return key


def deepgram_transcribe(audio_path):
    url = "https://api.deepgram.com/v1/listen"

    params = {
        "model": "nova-3",
        "detect_language": "true",
        "punctuate": "true",
        "smart_format": "true",
        "utterances": "true",
        "diarize": "true",
    }

    headers = {
        "Authorization": f"Token {deepgram_key()}",
        "Content-Type": "audio/wav",
    }

    data = audio_path.read_bytes()

    last_error = ""

    for attempt in range(3):
        try:
            response = requests.post(
                url,
                params=params,
                headers=headers,
                data=data,
                timeout=900,
            )

            if response.status_code == 200:
                obj = response.json()

                results = obj.get("results", {})

                utterances = results.get(
                    "utterances",
                    [],
                ) or []

                if utterances:
                    return utterances

                channels = results.get(
                    "channels",
                    [],
                ) or []

                if channels:
                    alternatives = channels[0].get(
                        "alternatives",
                        [],
                    ) or []

                    if alternatives:
                        words = alternatives[0].get(
                            "words",
                            [],
                        ) or []

                        if words:
                            return words_to_utterances(words)

                raise RuntimeError(
                    "Deepgram က transcript မပြန်ပေးပါ။"
                )

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

            if response.status_code not in (
                429,
                500,
                502,
                503,
                504,
            ):
                break

        except Exception as e:
            last_error = str(e)

        time.sleep(
            2 ** attempt + random.random()
        )

    raise RuntimeError(
        "Deepgram STT မအောင်မြင်ပါ။\n"
        + last_error
    )


def words_to_utterances(words):
    output = []
    current = []

    for word in words:
        current.append(word)

        punct = str(
            word.get(
                "punctuated_word",
                word.get("word", ""),
            )
        )

        if punct.endswith(
            (
                ".",
                "!",
                "?",
                "။",
                "！",
                "？",
            )
        ):
            output.append(
                {
                    "start": float(
                        current[0].get(
                            "start",
                            0,
                        )
                    ),
                    "end": float(
                        current[-1].get(
                            "end",
                            current[-1].get(
                                "start",
                                0,
                            ),
                        )
                    ),
                    "transcript": " ".join(
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
                    ).strip(),
                }
            )

            current = []

    if current:
        output.append(
            {
                "start": float(
                    current[0].get(
                        "start",
                        0,
                    )
                ),
                "end": float(
                    current[-1].get(
                        "end",
                        current[-1].get(
                            "start",
                            0,
                        ),
                    )
                ),
                "transcript": " ".join(
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
                ).strip(),
            }
        )

    return output


# =========================================================
# TEXT / TRANSLATION
# =========================================================

def clean_text(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def clean_json(text):
    text = (text or "").strip()

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
        return text[start:end + 1]

    return text


def translate_batch(client, items):
    prompt = (
        "You are a professional movie subtitle translator.\n"
        "Translate each source dialogue into natural "
        "conversational Burmese (Myanmar language).\n"
        "Keep names, meaning, emotion, and context.\n"
        "Do not add explanations.\n"
        "Keep each translation concise enough for the "
        "same timestamp.\n"
        "Return ONLY a JSON array with exactly the same "
        "number of objects and the same id values.\n"
        'Format: [{"id":1,"burmese":"..."}]\n\n'
        "INPUT:\n"
        + json.dumps(
            items,
            ensure_ascii=False,
        )
    )

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
                    clean_json(raw)
                )

                if (
                    not isinstance(data, list)
                    or len(data) != len(items)
                ):
                    raise RuntimeError(
                        "Gemini translation result "
                        "count မကိုက်ပါ။"
                    )

                by_id = {}

                for item in data:
                    item_id = int(item["id"])
                    Burmese = clean_text(
                        item["burmese"]
                    )
                    by_id[item_id] = Burmese

                expected_ids = range(
                    1,
                    len(items) + 1,
                )

                for item_id in expected_ids:
                    if (
                        item_id not in by_id
                        or not by_id[item_id]
                    ):
                        raise RuntimeError(
                            "ဘာသာပြန်စာကြောင်းတချို့ "
                            "မထွက်ပါ။"
                        )

                return by_id

            except Exception as e:
                error_text = str(e)

                errors.append(
                    f"{model} attempt "
                    f"{attempt + 1}: {error_text}"
                )

                if (
                    attempt == 0
                    and any(
                        word in error_text.lower()
                        for word in RETRY_WORDS
                    )
                ):
                    time.sleep(
                        3 + random.random() * 2
                    )
                else:
                    break

    raise RuntimeError(
        "Gemini translation မအောင်မြင်ပါ။\n"
        + "\n".join(errors[-8:])
    )


def build_segments(client, utterances, progress):
    rows = []

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
            and end > start + 0.05
        ):
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "source": text,
                }
            )

    if not rows:
        raise RuntimeError(
            "ပြောဆိုချက် မတွေ့ပါ။"
        )

    output = []

    batch_size = 12

    for position in range(
        0,
        len(rows),
        batch_size,
    ):
        batch = rows[
            position:
            position + batch_size
        ]

        payload = []

        for index, item in enumerate(
            batch,
            1,
        ):
            payload.append(
                {
                    "id": index,
                    "text": item["source"],
                }
            )

        progress(
            0.2
            + 0.45
            * (
                position
                / max(len(rows), 1)
            ),
            (
                "Gemini ဘာသာပြန်နေသည်... "
                f"{min(position + len(batch), len(rows))}"
                f"/{len(rows)}"
            ),
        )

        translated = translate_batch(
            client,
            payload,
        )

        for index, item in enumerate(
            batch,
            1,
        ):
            output.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "burmese": translated[index],
                }
            )

    return output


# =========================================================
# SRT
# =========================================================

def srt_time(seconds):
    seconds = max(
        0.0,
        float(seconds),
    )

    milliseconds = int(
        round(seconds * 1000)
    )

    hours, remainder = divmod(
        milliseconds,
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

    for index, segment in enumerate(
        segments,
        1,
    ):
        blocks.append(
            f"{index}\n"
            f"{srt_time(segment['start'])} --> "
            f"{srt_time(segment['end'])}\n"
            f"{segment['burmese']}\n"
        )

    return "\n".join(blocks)


def parse_srt(text):
    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )

    blocks = re.split(
        r"\n\s*\n",
        text,
    )

    output = []

    for block in blocks:
        lines = [
            line.strip("\ufeff")
            for line in block.split("\n")
        ]

        if len(lines) < 3:
            continue

        time_line = next(
            (
                line
                for line in lines
                if "-->" in line
            ),
            None,
        )

        if not time_line:
            continue

        match = re.match(
            r"\s*"
            r"(\d{2}:\d{2}:\d{2}[,.]\d{1,3})"
            r"\s*-->\s*"
            r"(\d{2}:\d{2}:\d{2}[,.]\d{1,3})",
            time_line,
        )

        if not match:
            continue

        def parse_time(value):
            value = value.replace(
                ",",
                ".",
            )

            hours, minutes, seconds = (
                value.split(":")
            )

            return (
                int(hours) * 3600
                + int(minutes) * 60
                + float(seconds)
            )

        start = parse_time(
            match.group(1)
        )

        end = parse_time(
            match.group(2)
        )

        time_index = lines.index(
            time_line
        )

        subtitle_lines = lines[
            time_index + 1:
        ]

        subtitle = clean_text(
            " ".join(subtitle_lines)
        )

        if (
            subtitle
            and end > start
        ):
            output.append(
                {
                    "start": start,
                    "end": end,
                    "burmese": subtitle,
                }
            )

    output.sort(
        key=lambda item: item["start"]
    )

    # Automatic overlap correction
    cleaned = []

    for item in output:
        if item["end"] <= item["start"]:
            continue

        if (
            cleaned
            and item["start"]
            < cleaned[-1]["end"]
        ):
            item["start"] = (
                cleaned[-1]["end"]
            )

        if item["end"] > item["start"]:
            cleaned.append(item)

    if not cleaned:
        raise RuntimeError(
            "SRT ထဲမှာ valid subtitle "
            "မတွေ့ပါ။"
        )

    return cleaned


# =========================================================
# BURMESE TTS
# =========================================================

async def edge_tts_save(
    text,
    voice,
    rate,
    pitch,
    output,
):
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=f"{rate:+d}%",
        pitch=f"{pitch:+d}Hz",
    )

    await communicate.save(
        str(output)
    )


def make_tts(
    text,
    voice,
    style,
    output,
):
    config = VOICE_STYLES[style]

    errors = []

    for attempt in range(3):
        try:
            if output.exists():
                output.unlink()

            asyncio.run(
                edge_tts_save(
                    text,
                    voice,
                    config["rate"],
                    config["pitch"],
                    output,
                )
            )

            if (
                output.exists()
                and output.stat().st_size > 1000
            ):
                return

            raise RuntimeError(
                "TTS file အလွတ်ဖြစ်နေပါသည်။"
            )

        except Exception as e:
            errors.append(str(e))

            time.sleep(
                2 + attempt
            )

    raise RuntimeError(
        "Burmese TTS မအောင်မြင်ပါ။\n"
        + "\n".join(errors[-3:])
    )


# =========================================================
# AUDIO SPEED / TIMING
# =========================================================

def atempo_chain(speed):
    speed = max(
        0.25,
        min(float(speed), 4.0),
    )

    parts = []

    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0

    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5

    parts.append(
        f"atempo={speed:.6f}"
    )

    return ",".join(parts)


def fit_clip(
    source,
    output,
    slot,
    user_speed,
):
    raw_duration = ffprobe_duration(
        source
    )

    desired_speed = max(
        0.25,
        min(float(user_speed), 2.0),
    )

    final_speed = (
        raw_duration
        / max(slot, 0.05)
    ) * desired_speed

    final_speed = max(
        0.25,
        min(final_speed, 4.0),
    )

    filter_audio = (
        atempo_chain(final_speed)
        + ",apad,atrim=duration="
        + f"{slot:.3f}"
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
            filter_audio,
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
            "Voiceover timing ပြင်မရပါ။\n"
            + (result.stderr or "")
        )


# =========================================================
# BUILD COMPLETE VOICEOVER
# =========================================================

def build_voiceover(
    segments,
    voice,
    style,
    speed,
    work,
    progress,
):
    clips = []

    total = len(segments)

    for index, segment in enumerate(
        segments,
        1,
    ):
        start = max(
            0.0,
            float(segment["start"]),
        )

        end = max(
            start + 0.05,
            float(segment["end"]),
        )

        slot = end - start

        raw_audio = (
            work
            / f"tts_{index:04d}.mp3"
        )

        fitted_audio = (
            work
            / f"clip_{index:04d}.m4a"
        )

        progress(
            0.1
            + 0.65
            * (
                (index - 1)
                / max(total, 1)
            ),
            f"Voice {index}/{total} ထုတ်နေသည်...",
        )

        make_tts(
            segment["burmese"],
            voice,
            style,
            raw_audio,
        )

        fit_clip(
            raw_audio,
            fitted_audio,
            slot,
            speed,
        )

        clips.append(
            (
                start,
                fitted_audio,
            )
        )

    output = (
        work
        / "burmese_voiceover.m4a"
    )

    command = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    for _, clip in clips:
        command += [
            "-i",
            str(clip),
        ]

    filters = []
    labels = []

    for index, (
        start,
        _,
    ) in enumerate(clips):
        milliseconds = max(
            0,
            int(round(start * 1000)),
        )

        label = f"v{index}"

        filters.append(
            f"[{index}:a]"
            f"adelay={milliseconds}:all=1,"
            f"aresample=48000"
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
        "alimiter=limit=0.95"
        "[out]"
    )

    command += [
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
        str(output),
    ]

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
            "Voiceover file မထုတ်နိုင်ပါ။\n"
            + (result.stderr or "")
        )

    # Final audio validation
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
            "Voiceover audio validation "
            "မအောင်မြင်ပါ။\n"
            + (check.stderr or "")
        )

    progress(
        1.0,
        "Voiceover ပြီးပါပြီ",
    )

    return output


# =========================================================
# HERO
# =========================================================

st.markdown(
"""<div class="hero">
<h1>🎬 Myanmar Movie AI</h1>
<p>Chinese / foreign movie dialogue → 🇲🇲 Natural Myanmar SRT → 🎙️ Burmese Voiceover</p>
<div style="margin-top:12px">
<span class="badge">🤖 AI Translation</span>
<span class="badge">⏱️ Auto Timing Check</span>
<span class="badge">🎙️ 2 Burmese Voices</span>
<span class="badge">📱 Mobile Friendly</span>
</div>
</div>""",
    unsafe_allow_html=True,
)


# =========================================================
# STEP 1
# =========================================================

st.markdown(
    '<div class="step">'
    '① 🎥 Video → Myanmar SRT'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "AI က dialogue, timestamp နဲ့ translation "
    "ကို အလိုအလျောက်လုပ်ပေးမယ်။"
)

video_file = st.file_uploader(
    "🎥 Movie / Video တင်ပါ",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm",
    ],
    key="video",
)

make_srt_button = st.button(
    "🚀 AI SRT စတင်ထုတ်မယ်",
    type="primary",
    use_container_width=True,
)


if make_srt_button:

    if not video_file:
        st.error(
            "Video တစ်ခုအရင်တင်ပါ။"
        )
        st.stop()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            work = Path(temp_dir)

            video = (
                work
                / "input_video"
            )

            audio = (
                work
                / "audio.wav"
            )

            video.write_bytes(
                video_file.getbuffer()
            )

            status = st.empty()
            progress_bar = st.progress(0.0)

            status.info(
                "🎧 1/3 Audio ကို စစ်နေသည်..."
            )

            extract_audio(
                video,
                audio,
            )

            progress_bar.progress(
                0.15
            )

            status.info(
                "🗣️ 2/3 Dialogue + timestamp "
                "ရယူနေသည်..."
            )

            utterances = (
                deepgram_transcribe(
                    audio
                )
            )

            progress_bar.progress(
                0.30
            )

            status.info(
                "🇲🇲 3/3 Gemini က "
                "သဘာဝကျမြန်မာလို "
                "ဘာသာပြန်နေသည်..."
            )

            segments = build_segments(
                gemini_client(),
                utterances,
                lambda p, t: (
                    progress_bar.progress(
                        min(p, 0.9)
                    ),
                    status.info(t),
                ),
            )

            srt = make_srt(
                segments
            )

            st.session_state[
                "srt_text"
            ] = srt

            st.session_state[
                "srt_name"
            ] = (
                Path(
                    video_file.name
                ).stem
                + "_myanmar.srt"
            )

            st.session_state[
                "srt_meta"
            ] = {
                "lines": len(segments),
                "file": video_file.name,
            }

            progress_bar.progress(
                1.0
            )

            status.success(
                "✅ SRT အောင်မြင်ပါပြီ — "
                f"{len(segments)} dialogue lines"
            )

    except Exception as error:
        st.error(
            "SRT ထုတ်ရာမှာ "
            "အမှားဖြစ်ပါတယ်။"
        )

        st.exception(error)


# =========================================================
# SRT RESULT
# =========================================================

if st.session_state.get(
    "srt_text"
):

    metadata = st.session_state.get(
        "srt_meta",
        {},
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "📝 Dialogue",
        f"{metadata.get('lines', 0)} lines",
    )

    col2.metric(
        "🤖 Status",
        "AI Checked",
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.subheader(
        "📄 Myanmar SRT Preview"
    )

    st.text_area(
        "Preview",
        st.session_state[
            "srt_text"
        ],
        height=300,
        disabled=True,
        label_visibility="collapsed",
    )

    st.download_button(
        "⬇️ Download Myanmar SRT",
        st.session_state[
            "srt_text"
        ].encode("utf-8-sig"),
        st.session_state.get(
            "srt_name",
            "myanmar.srt",
        ),
        "application/x-subrip",
        use_container_width=True,
    )


# =========================================================
# STEP 2
# =========================================================

st.markdown(
    '<div class="step">'
    '② 🎙️ SRT → Myanmar Voiceover'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Voice, style နဲ့ speed ကိုရွေးပါ။ "
    "Timing / overlap ကို app က "
    "အလိုအလျောက်စစ်ပြီးညှိပေးမယ်။"
)

srt_file = st.file_uploader(
    "📄 SRT တင်ပါ "
    "(သို့) အဆင့် ၁ ရဲ့ SRT ကိုသုံးပါ",
    type=["srt"],
    key="srt_upload",
)


voice_col, style_col = st.columns(2)

with voice_col:

    voice_name = st.selectbox(
        "🎙️ Voice",
        list(VOICES.keys()),
        key="voice_choice",
    )

with style_col:

    style = st.selectbox(
        "🎭 Voice Style",
        list(VOICE_STYLES.keys()),
        key="style_choice",
    )


speed = st.slider(
    "⚡ Speaking Speed",
    0.70,
    1.30,
    1.00,
    0.05,
    help=(
        "နောက်ဆုံး audio timing ကို "
        "SRT slot ထဲဝင်အောင် app က "
        "auto-fit လုပ်ပေးမယ်။"
    ),
)


voice_description = {
    "သီဟ (အမျိုးသား)":
        "👨 Male • clear / natural",

    "နီလာ (အမျိုးသမီး)":
        "👩 Female • clear / natural",
}


st.info(
    f"🎙️ Selected: **"
    f"{voice_description[voice_name]}"
    f"**  •  🎭 **{style}**  •  "
    f"⚡ **{speed:.2f}x**"
)


make_voice_button = st.button(
    "🎧 Voiceover စတင်ထုတ်မယ်",
    type="primary",
    use_container_width=True,
)


# =========================================================
# VOICEOVER PROCESS
# =========================================================

if make_voice_button:

    source_srt = None

    if srt_file:

        source_srt = (
            srt_file
            .getvalue()
            .decode(
                "utf-8-sig",
                errors="replace",
            )
        )

    elif st.session_state.get(
        "srt_text"
    ):

        source_srt = (
            st.session_state[
                "srt_text"
            ]
        )

    else:

        st.error(
            "SRT ဖိုင်တင်ပါ "
            "(သို့) အဆင့် ၁ မှာ "
            "SRT အရင်ထုတ်ပါ။"
        )

        st.stop()

    try:

        segments = parse_srt(
            source_srt
        )

        with tempfile.TemporaryDirectory() as temp_dir:

            work = Path(temp_dir)

            status = st.empty()
            progress_bar = st.progress(0.0)

            status.info(
                "🔎 AI/App က SRT timing "
                "စစ်နေသည်... "
                f"{len(segments)} lines"
            )

            voiceover = build_voiceover(
                segments,
                VOICES[voice_name],
                style,
                speed,
                work,
                lambda p, t: (
                    progress_bar.progress(
                        min(p, 1.0)
                    ),
                    status.info(t),
                ),
            )

            voice_bytes = (
                voiceover.read_bytes()
            )

            st.session_state[
                "voice_bytes"
            ] = voice_bytes

            st.session_state[
                "voice_name"
            ] = (
                "myanmar_voiceover.m4a"
            )

            st.session_state[
                "voice_mime"
            ] = "audio/mp4"

            st.session_state[
                "voice_segments"
            ] = segments

            progress_bar.progress(
                1.0
            )

            status.success(
                "✅ Voiceover ပြီးပါပြီ — "
                "timing validation "
                "လုပ်ပြီးပါပြီ။"
            )

    except Exception as error:

        st.error(
            "Voiceover ထုတ်ရာမှာ "
            "အမှားဖြစ်ပါတယ်။"
        )

        st.exception(error)


# =========================================================
# VOICEOVER RESULT
# =========================================================

if st.session_state.get(
    "voice_bytes"
):

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    st.subheader(
        "🔊 Voiceover Preview"
    )

    st.audio(
        st.session_state[
            "voice_bytes"
        ],
        format=st.session_state.get(
            "voice_mime",
            "audio/mp4",
        ),
    )

    st.download_button(
        "⬇️ Download Voiceover",
        st.session_state[
            "voice_bytes"
        ],
        st.session_state.get(
            "voice_name",
            "myanmar_voiceover.m4a",
        ),
        st.session_state.get(
            "voice_mime",
            "audio/mp4",
        ),
        use_container_width=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🔒 Core processing ကိုမပြောင်းထားပါ။ "
    "ဒီ version မှာ video merge / "
    "original audio removal မပါပါ။ "
    "Voice Clone ကို နောက် Phase မှာ "
    "သီးခြားထည့်နိုင်ပါတယ်။"
)
