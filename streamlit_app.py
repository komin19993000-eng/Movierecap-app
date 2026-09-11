import asyncio
import json
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

import edge_tts
import imageio_ffmpeg
import requests
import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Myanmar Movie AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

VOICES = {
    "သီဟ (အမျိုးသား)": "my-MM-ThihaNeural",
    "နီလာ (အမျိုးသမီး)": "my-MM-NilarNeural",
}

VOICE_STYLES = {
    "ပုံမှန်": {"rate": 0, "pitch": 0},
    "နက်နက် (Deep)": {"rate": -5, "pitch": -12},
    "ပျော့ပျောင်း": {"rate": -3, "pitch": 5},
    "တက်ကြွ": {"rate": 8, "pitch": 2},
}

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

RETRY_STATUS = {429, 500, 502, 503, 504}

MAX_SOURCE_CHARS = 42
MAX_BURMESE_CHARS = 50
MAX_SEGMENT_DURATION = 6.0
PAUSE_SPLIT = 0.65

MAX_TTS_SPEEDUP = 1.15
MIN_TTS_SPEED = 0.80
MAX_TTS_SPEED = 1.30

VOICE_GAP = 0.04


# ============================================================
# UI STYLE
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 5%,
            rgba(255, 0, 128, 0.22),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 5%,
            rgba(0, 220, 255, 0.22),
            transparent 28%
        ),
        radial-gradient(
            circle at 50% 95%,
            rgba(130, 60, 255, 0.20),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            #f7fbff 0%,
            #fff5fb 45%,
            #f4f9ff 100%
        );

    color: #000000;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.stApp,
.stApp p,
.stApp label,
.stApp span,
.stApp div,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #000000;
}

.hero-box {
    padding: 28px 24px;
    margin-bottom: 26px;
    border-radius: 26px;

    background:
        linear-gradient(
            135deg,
            #00e5ff 0%,
            #6c3cff 45%,
            #ff299c 100%
        );

    border: 3px solid #000000;

    box-shadow:
        0 10px 0 #000000,
        0 18px 35px rgba(110, 40, 180, 0.30);

    text-align: center;
}

.hero-title {
    margin: 0;
    color: #000000 !important;
    font-size: clamp(30px, 7vw, 52px);
    font-weight: 900;
    letter-spacing: -1px;
}

.hero-subtitle {
    margin-top: 8px;
    color: #000000 !important;
    font-size: 16px;
    font-weight: 700;
}

.section-title {
    margin-top: 28px;
    margin-bottom: 14px;

    padding: 14px 18px;

    border-radius: 18px;

    background:
        linear-gradient(
            90deg,
            #00e5ff,
            #7b3cff,
            #ff299c
        );

    border: 3px solid #000000;

    box-shadow: 0 6px 0 #000000;

    color: #000000 !important;

    font-size: 25px;
    font-weight: 900;
}

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 2px solid #000000 !important;
    border-radius: 13px !important;
}

input,
textarea {
    color: #000000 !important;
    background: #ffffff !important;
    -webkit-text-fill-color: #000000 !important;
}

textarea {
    border-radius: 14px !important;
}

div[data-testid="stFileUploader"] {
    background:
        linear-gradient(
            135deg,
            rgba(0, 229, 255, 0.22),
            rgba(255, 41, 156, 0.18)
        );

    border: 3px solid #000000;
    border-radius: 18px;
    padding: 10px;
}

div[data-testid="stFileUploader"] section {
    background: #ffffff !important;
    border-radius: 13px !important;
}

div[data-testid="stFileUploader"] button {
    color: #000000 !important;
    background: #ffffff !important;
    border: 2px solid #000000 !important;
}

div[data-baseweb="select"] * {
    color: #000000 !important;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button,
button[kind="primary"] {

    min-height: 52px;

    color: #000000 !important;

    background:
        linear-gradient(
            90deg,
            #00e5ff 0%,
            #6c3cff 50%,
            #ff299c 100%
        ) !important;

    border: 3px solid #000000 !important;

    border-radius: 15px !important;

    font-size: 16px !important;
    font-weight: 900 !important;

    box-shadow: 0 5px 0 #000000 !important;

    transition: all 0.12s ease;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 0 #000000 !important;
}

div.stButton > button:active,
div[data-testid="stFormSubmitButton"] button:active {
    transform: translateY(3px);
    box-shadow: 0 2px 0 #000000 !important;
}

div[data-testid="stDownloadButton"] button {

    min-height: 52px;

    color: #000000 !important;

    background:
        linear-gradient(
            90deg,
            #00e5ff,
            #7b3cff,
            #ff299c
        ) !important;

    border: 3px solid #000000 !important;

    border-radius: 15px !important;

    font-size: 16px !important;

    font-weight: 900 !important;

    box-shadow: 0 5px 0 #000000 !important;
}

div[data-testid="stProgress"] > div {
    background: #ffffff !important;
    border: 2px solid #000000;
    border-radius: 20px;
}

div[data-testid="stProgress"] div[role="progressbar"] {
    background:
        linear-gradient(
            90deg,
            #00e5ff,
            #7b3cff,
            #ff299c
        ) !important;
}

div[data-testid="stAlert"] {
    border: 2px solid #000000 !important;
    border-radius: 14px !important;
}

div[data-testid="stAlert"] * {
    color: #000000 !important;
}

.srt-title {
    margin-top: 22px;
    margin-bottom: 12px;

    color: #000000 !important;

    font-size: 25px;
    font-weight: 900;
}

hr {
    border: 0 !important;
    height: 5px !important;

    background:
        linear-gradient(
            90deg,
            #00e5ff,
            #7b3cff,
            #ff299c
        ) !important;

    border-radius: 10px;
    margin: 34px 0 !important;
}

.footer {
    text-align: center;
    color: #000000 !important;
    font-size: 13px;
    font-weight: 700;
    padding-top: 14px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero-box">
    <div class="hero-title">🎬 Myanmar Movie AI</div>
    <div class="hero-subtitle">
        Video → မြန်မာ SRT → Burmese Voiceover
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""

    if value:
        return str(value).strip()

    return os.getenv(name, "").strip()


def safe_filename(name: str, default: str) -> str:
    value = str(name or "").strip()

    value = re.sub(
        r'[\\/:*?"<>|]+',
        "_",
        value,
    )

    value = re.sub(
        r"\s+",
        "_",
        value,
    )

    return value or default


def clean_text(text: str) -> str:
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


def ffprobe_duration(path: Path) -> float:
    result = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(path),
        ],
        timeout=120,
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr or "",
    )

    if not match:
        raise RuntimeError(
            "Media duration ကို ဖတ်မရပါ။"
        )

    return (
        int(match.group(1)) * 3600
        + int(match.group(2)) * 60
        + float(match.group(3))
    )


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio(
    video_path: Path,
    audio_path: Path,
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
# GEMINI (WITH KEY ROTATION)
# ============================================================

def get_gemini_keys() -> list[str]:
    keys = []

    main_key = get_secret("GEMINI_API_KEY")

    if main_key:
        for k in main_key.split(","):
            k = k.strip()

            if k and k not in keys:
                keys.append(k)

    for i in range(1, 10):

        key = get_secret(
            f"GEMINI_API_KEY_{i}"
        )

        if key and key not in keys:
            keys.append(key)

    if not keys:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return keys


def get_gemini_client(
    key_index: int = 0
):
    keys = get_gemini_keys()

    selected_key = keys[
        key_index % len(keys)
    ]

    return genai.Client(
        api_key=selected_key
    )


def get_gemini_model() -> str:
    return (
        get_secret("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    )


def extract_json_array(text: str):

    value = (
        text or ""
    ).strip()

    value = re.sub(
        r"^```(?:json)?\s*",
        "",
        value,
        flags=re.I,
    )

    value = re.sub(
        r"\s*```$",
        "",
        value,
    )

    start = value.find("[")
    end = value.rfind("]")

    if start < 0 or end <= start:
        raise ValueError(
            "Gemini က JSON result မပြန်ပေးပါ။"
        )

    return json.loads(
        value[start:end + 1]
    )


def shorten_burmese_text(
    text: str
):

    text = clean_text(text)

    if len(text) <= MAX_BURMESE_CHARS:
        return text

    punctuation_positions = []

    for mark in [
        "။",
        "၊",
        ",",
        ".",
        "!",
        "?",
        "…",
    ]:

        pos = text.rfind(
            mark,
            0,
            MAX_BURMESE_CHARS + 1,
        )

        if pos >= 20:

            punctuation_positions.append(
                pos + 1
            )

    if punctuation_positions:

        return clean_text(
            text[
                :max(
                    punctuation_positions
                )
            ]
        )

    words = text.split()

    result = []
    length = 0

    for word in words:

        extra = len(word) + (
            1 if result else 0
        )

        if (
            length + extra
            > MAX_BURMESE_CHARS
        ):
            break

        result.append(word)
        length += extra

    shortened = clean_text(
        " ".join(result)
    )

    if shortened:
        return shortened

    return text[
        :MAX_BURMESE_CHARS
    ].strip()


def translate_batch(
    client_ignored,
    rows
):

    payload = [
        {
            "id": i + 1,
            "text": row["source"],
            "duration": round(
                row["end"]
                - row["start"],
                2,
            ),
        }
        for i, row
        in enumerate(rows)
    ]

    prompt = f"""
You are a professional Myanmar movie dubbing translator.

Translate every dialogue into natural spoken Burmese
that sounds like a real Myanmar movie dub.

IMPORTANT:
The Burmese sentence will be spoken by TTS.
Therefore it MUST be concise and easy to speak naturally.

Rules:
- Preserve the original meaning.
- Preserve names and important proper nouns.
- Preserve emotion, intention and tone.
- Do NOT summarize away important meaning.
- Do NOT add explanations.
- Do NOT add quotation marks unless required by meaning.
- Do NOT translate word-for-word if that sounds unnatural.
- Use natural conversational Burmese.
- Avoid unnecessary filler words.
- Avoid repeating information.
- Prefer shorter natural Burmese wording.
- Aim for approximately 25–50 Burmese characters.
- Never intentionally create a very long sentence.
- The subtitle must be suitable for dubbing within its timestamp.
- Return ONLY JSON.
- Return exactly {len(payload)} objects.
- Keep exact id values.

Format:
[
  {{"id":1,"burmese":"မြန်မာစာ"}}
]

INPUT:
{json.dumps(
    payload,
    ensure_ascii=False
)}
"""

    model = get_gemini_model()
    keys = get_gemini_keys()

    if (
        "current_key_index"
        not in st.session_state
    ):
        st.session_state.current_key_index = 0

    total_keys = len(keys)

    attempts = 0
    max_attempts = total_keys * 3

    last_error = ""

    while attempts < max_attempts:

        current_idx = (
            st.session_state.current_key_index
            % total_keys
        )

        client = get_gemini_client(
            current_idx
        )

        try:

            response = (
                client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=(
                        types.GenerateContentConfig(
                            response_mime_type=(
                                "application/json"
                            ),
                        )
                    ),
                )
            )

            data = extract_json_array(
                getattr(
                    response,
                    "text",
                    ""
                )
            )

            if (
                not isinstance(
                    data,
                    list
                )
                or len(data)
                != len(payload)
            ):

                raise RuntimeError(
                    "Gemini translation result count မကိုက်ပါ။"
                )

            translated = {}

            for item in data:

                idx = int(
                    item["id"]
                )

                text = clean_text(
                    item.get(
                        "burmese",
                        ""
                    )
                )

                if not text:

                    raise RuntimeError(
                        f"Subtitle {idx} အတွက် "
                        "ဘာသာပြန်စာ မထွက်ပါ။"
                    )

                text = shorten_burmese_text(
                    text
                )

                translated[idx] = text

            if (
                len(translated)
                != len(payload)
            ):

                raise RuntimeError(
                    "ဘာသာပြန်စာကြောင်းတချို့ မထွက်ပါ။"
                )

            return translated

        except Exception as exc:

            last_error = str(exc)

            low = last_error.lower()

            transient = any(
                word in low
                for word in [
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "timeout",
                    "unavailable",
                    "overloaded",
                    "resource exhausted",
                    "high demand",
                    "quota",
                ]
            )

            if (
                transient
                and total_keys > 1
            ):

                st.session_state.current_key_index = (
                    (
                        st.session_state.current_key_index
                        + 1
                    )
                    % total_keys
                )

                attempts += 1

                time.sleep(
                    1
                    + random.random()
                )

            elif (
                transient
                and attempts < 2
            ):

                attempts += 1

                time.sleep(
                    (2 ** attempts)
                    + random.random()
                )

            else:

                break

    raise RuntimeError(
        "Gemini translation မအောင်မြင်ပါ။\n"
        + last_error
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
            "Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return key


def words_to_segments(words):

    max_chars = MAX_SOURCE_CHARS
    max_duration = MAX_SEGMENT_DURATION
    pause_split = PAUSE_SPLIT

    punctuation = (
        ".",
        "!",
        "?",
        "။",
        "！",
        "？",
    )

    segments = []
    current = []

    def word_text(item):

        return str(
            item.get(
                "punctuated_word"
            )
            or item.get(
                "word"
            )
            or ""
        ).strip()

    def flush():

        nonlocal current

        if not current:
            return

        first = current[0]
        last = current[-1]

        start = float(
            first.get(
                "start",
                0.0
            )
        )

        end = float(
            last.get(
                "end",
                start
            )
        )

        text = clean_text(
            " ".join(
                word_text(x)
                for x in current
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

        if not word_text(word):
            continue

        if current:

            previous_end = float(
                current[-1].get(
                    "end",
                    current[-1].get(
                        "start",
                        0.0,
                    ),
                )
            )

            current_start = float(
                word.get(
                    "start",
                    previous_end,
                )
            )

            proposed = clean_text(
                " ".join(
                    word_text(x)
                    for x
                    in current + [word]
                )
            )

            proposed_end = float(
                word.get(
                    "end",
                    current_start,
                )
            )

            duration = (
                proposed_end
                - float(
                    current[0].get(
                        "start",
                        current_start,
                    )
                )
            )

            pause = (
                current_start
                - previous_end
            )

            if (
                pause >= pause_split
                or len(proposed)
                > max_chars
                or duration
                > max_duration
            ):

                flush()

        current.append(word)

        if word_text(word).endswith(
            punctuation
        ):

            flush()

    flush()

    return segments


def deepgram_transcribe(
    audio_path: Path,
):

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
        "Authorization":
            f"Token {get_deepgram_key()}",
        "Content-Type":
            "audio/wav",
    }

    audio_data = (
        audio_path.read_bytes()
    )

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

                obj = response.json()

                results = obj.get(
                    "results",
                    {},
                )

                channels = results.get(
                    "channels",
                    [],
                ) or []

                words = []

                if channels:

                    alternatives = (
                        channels[0].get(
                            "alternatives",
                            [],
                        )
                        or []
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
                            0.0,
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
                    "Deepgram က transcript "
                    "မပြန်ပေးပါ။"
                )

            last_error = (
                f"HTTP "
                f"{response.status_code}: "
                f"{response.text[:700]}"
            )

            if (
                response.status_code
                not in RETRY_STATUS
            ):

                break

        except Exception as exc:

            last_error = str(exc)

        if attempt < 2:

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

def srt_time(
    seconds: float
) -> str:

    total_ms = max(
        0,
        int(
            round(
                float(seconds)
                * 1000
            )
        ),
    )

    hours, rem = divmod(
        total_ms,
        3600000,
    )

    minutes, rem = divmod(
        rem,
        60000,
    )

    seconds, millis = divmod(
        rem,
        1000,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d},"
        f"{millis:03d}"
    )


def build_srt_segments(
    client,
    source_segments,
    progress_callback,
):

    rows = []

    for item in source_segments:

        text = clean_text(
            item.get(
                "source",
                ""
            )
        )

        start = float(
            item.get(
                "start",
                0.0
            )
        )

        end = float(
            item.get(
                "end",
                start
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
            "Dialogue မတွေ့ပါ။"
        )

    result = []

    batch_size = 10
    total = len(rows)

    for pos in range(
        0,
        total,
        batch_size,
    ):

        batch = rows[
            pos:pos + batch_size
        ]

        translated = (
            translate_batch(
                client,
                batch,
            )
        )

        for local_index, row in enumerate(
            batch,
            start=1,
        ):

            burmese = translated[
                local_index
            ]

            result.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "burmese": burmese,
                }
            )

        progress_callback(
            min(
                0.95,
                0.25
                + 0.70
                * (
                    (
                        pos
                        + len(batch)
                    )
                    / total
                ),
            ),
            (
                "ဘာသာပြန်ပြီးပါပြီ — "
                f"{min(pos + len(batch), total)}"
                f"/{total}"
            ),
        )

    return result


def make_srt(
    segments
):

    blocks = []

    for index, item in enumerate(
        segments,
        start=1,
    ):

        text = clean_text(
            item["burmese"]
        )

        blocks.append(
            f"{index}\n"
            f"{srt_time(item['start'])} --> "
            f"{srt_time(item['end'])}\n"
            f"{text}\n"
        )

    return "\n".join(blocks)


def parse_srt_timestamp(
    value: str,
) -> float:

    value = value.replace(
        ",",
        ".",
    ).strip()

    parts = value.split(":")

    if len(parts) != 3:

        raise ValueError(
            "Invalid timestamp"
        )

    h, m, s = parts

    return (
        int(h) * 3600
        + int(m) * 60
        + float(s)
    )


def parse_srt(
    text: str
):

    normalized = (
        str(text or "")
        .replace(
            "\r\n",
            "\n"
        )
        .replace(
            "\r",
            "\n"
        )
        .strip("\ufeff \n")
    )

    blocks = re.split(
        r"\n\s*\n",
        normalized,
    )

    entries = []

    for block in blocks:

        lines = [
            line.strip("\ufeff")
            for line
            in block.split("\n")
        ]

        time_index = next(
            (
                i
                for i, line
                in enumerate(lines)
                if "-->" in line
            ),
            None,
        )

        if time_index is None:
            continue

        match = re.match(
            r"^\s*"
            r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
            r"\s*-->\s*"
            r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})",
            lines[time_index],
        )

        if not match:
            continue

        try:

            start = (
                parse_srt_timestamp(
                    match.group(1)
                )
            )

            end = (
                parse_srt_timestamp(
                    match.group(2)
                )
            )

        except Exception:

            continue

        subtitle = clean_text(
            " ".join(
                x.strip()
                for x in lines[
                    time_index + 1:
                ]
                if x.strip()
            )
        )

        if (
            subtitle
            and end > start
        ):

            entries.append(
                {
                    "start": start,
                    "end": end,
                    "burmese": subtitle,
                }
            )

    if not entries:

        raise RuntimeError(
            "SRT ထဲမှာ valid subtitle မတွေ့ပါ။"
        )

    entries.sort(
        key=lambda item:
            item["start"]
    )

    cleaned = []
    fixed = 0

    for item in entries:

        start = item["start"]
        end = item["end"]

        if (
            cleaned
            and start
            < cleaned[-1]["end"]
        ):

            start = (
                cleaned[-1]["end"]
            )

            fixed += 1

        if end > start:

            cleaned.append(
                {
                    "start": start,
                    "end": end,
                    "burmese": item[
                        "burmese"
                    ],
                }
            )

        else:

            fixed += 1

    if not cleaned:

        raise RuntimeError(
            "SRT timing ပြင်ပြီးနောက် "
            "valid subtitle မကျန်ပါ။"
        )

    return cleaned, fixed


# ============================================================
# TTS
# ============================================================

async def edge_tts_save(
    text,
    voice,
    rate,
    pitch,
    output_path,
):

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=f"{int(rate):+d}%",
        pitch=f"{int(pitch):+d}Hz",
    )

    await communicate.save(
        str(output_path)
    )


def make_tts(
    text,
    voice,
    style,
    output_path,
):

    settings = VOICE_STYLES[
        style
    ]

    errors = []

    for attempt in range(3):

        try:

            if output_path.exists():
                output_path.unlink()

            asyncio.run(
                edge_tts_save(
                    text,
                    voice,
                    settings["rate"],
                    settings["pitch"],
                    output_path,
                )
            )

            if (
                output_path.exists()
                and output_path.stat().st_size
                > 1000
            ):

                return

            raise RuntimeError(
                "TTS file အလွတ်ဖြစ်နေပါသည်။"
            )

        except Exception as exc:

            errors.append(
                str(exc)
            )

            if attempt < 2:

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
# SAFE AUDIO SPEED
# ============================================================

def atempo_chain(
    factor: float,
) -> str:

    factor = max(
        MIN_TTS_SPEED,
        min(
            float(factor),
            MAX_TTS_SPEEDUP,
        ),
    )

    return f"atempo={factor:.6f}"


def fit_tts_to_slot(
    source: Path,
    output: Path,
    slot: float,
    user_speed: float,
):

    raw_duration = ffprobe_duration(
        source
    )

    slot = max(
        0.20,
        float(slot),
    )

    user_speed = max(
        MIN_TTS_SPEED,
        min(
            float(user_speed),
            MAX_TTS_SPEED,
        ),
    )

    required_factor = (
        raw_duration / slot
    )

    if required_factor <= 1.0:

        factor = user_speed

    else:

        factor = max(
            required_factor,
            user_speed,
        )

        factor = min(
            factor,
            MAX_TTS_SPEEDUP,
        )

    # IMPORTANT:
    # Do NOT use apad here.
    # apad can extend each clip beyond its subtitle timing
    # and make the final voiceover much longer than the video.
    #
    # Instead:
    # 1. Change speed when necessary.
    # 2. Keep the audio inside the exact subtitle slot.
    # 3. Trim the result to the slot duration.

    audio_filter = (
        atempo_chain(factor)
        + ",atrim=duration="
        + f"{slot:.3f}"
        + ",asetpts=PTS-STARTPTS"
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
        timeout=180,
    )

    if (
        result.returncode != 0
        or not output.exists()
        or output.stat().st_size < 1000
    ):

        raise RuntimeError(
            "Voiceover timing ပြင်မရပါ။\n"
            + (
                result.stderr
                or ""
            )
        )


# ============================================================
# VOICEOVER
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
        start=1,
    ):

        start = max(
            0.0,
            float(item["start"]),
        )

        current_end = max(
            start + 0.20,
            float(item["end"]),
        )

        if index < total:

            next_start = max(
                current_end,
                float(
                    segments[index]["start"]
                ),
            )

            available_slot = (
                next_start
                - start
                - VOICE_GAP
            )

            slot = max(
                current_end - start,
                available_slot,
            )

        else:

            slot = (
                current_end
                - start
            )

        slot = max(
            0.20,
            float(slot),
        )

        raw = (
            work_dir
            / f"tts_{index:04d}.mp3"
        )

        fitted = (
            work_dir
            / f"clip_{index:04d}.m4a"
        )

        progress_callback(
            0.05
            + 0.75
            * (
                (index - 1)
                / total
            ),
            (
                f"Voice {index}/{total} "
                "ထုတ်နေသည်..."
            ),
        )

        text = clean_text(
            item["burmese"]
        )

        make_tts(
            text,
            voice,
            style,
            raw,
        )

        fit_tts_to_slot(
            raw,
            fitted,
            slot,
            speed,
        )

        clips.append(
            (
                start,
                fitted,
            )
        )

    if not clips:

        raise RuntimeError(
            "Voiceover အတွက် subtitle မရှိပါ။"
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

    for _, clip in clips:

        command.extend(
            [
                "-i",
                str(clip),
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

        label = f"v{index}"

        filters.append(
            f"[{index}:a]"
            f"adelay={delay_ms}:all=1,"
            f"aresample=48000"
            f"[{label}]"
        )

        labels.append(
            f"[{label}]"
        )

    filters.append(
        "".join(labels)
        + f"amix="
        f"inputs={len(labels)}:"
        f"duration=longest:"
        f"dropout_transition=0,"
        "loudnorm=I=-16:"
        "TP=-1.5:"
        "LRA=11,"
        "volume=1.25,"
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
        timeout=1800,
    )

    if (
        result.returncode != 0
        or not output.exists()
        or output.stat().st_size < 5000
    ):

        raise RuntimeError(
            "Voiceover file မထုတ်နိုင်ပါ။\n"
            + (
                result.stderr
                or ""
            )
        )

    validation = run_cmd(
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
        timeout=300,
    )

    if validation.returncode != 0:

        raise RuntimeError(
            "Voiceover audio validation "
            "မအောင်မြင်ပါ။\n"
            + (
                validation.stderr
                or ""
            )
        )

    progress_callback(
        1.0,
        "Voiceover ပြီးပါပြီ",
    )

    return output


# ============================================================
# SESSION STATE
# ============================================================

if "srt_text" not in st.session_state:
    st.session_state.srt_text = ""

if "srt_name" not in st.session_state:
    st.session_state.srt_name = (
        "myanmar.srt"
    )

if "voice_bytes" not in st.session_state:
    st.session_state.voice_bytes = None

if "voice_name" not in st.session_state:
    st.session_state.voice_name = (
        "myanmar_voiceover.m4a"
    )

if "current_key_index" not in st.session_state:
    st.session_state.current_key_index = 0


# ============================================================
# STEP 1
# ============================================================

st.markdown(
    '<div class="section-title">'
    '① Video → မြန်မာ SRT'
    '</div>',
    unsafe_allow_html=True,
)

video_file = st.file_uploader(
    "🎥 Video တင်ပါ",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm",
    ],
    key="source_video",
)

with st.form(
    "srt_form",
    clear_on_submit=False,
):

    srt_output_name = st.text_input(
        "💾 SRT filename",
        value=(
            Path(
                video_file.name
            ).stem
            + "_myanmar.srt"
        )
        if video_file
        else "myanmar.srt",
    )

    make_srt_button = (
        st.form_submit_button(
            "📝 မြန်မာ SRT ထုတ်မယ်",
            type="primary",
            use_container_width=True,
        )
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

            video_path = (
                work
                / "input_video"
            )

            audio_path = (
                work
                / "audio.wav"
            )

            video_path.write_bytes(
                video_file.getbuffer()
            )

            status = st.empty()
            progress = st.progress(
                0.0
            )

            status.info(
                "🎧 Audio ထုတ်နေသည်..."
            )

            extract_audio(
                video_path,
                audio_path,
            )

            progress.progress(
                0.12
            )

            status.info(
                "🎙️ Dialogue timestamp ရယူနေသည်..."
            )

            source_segments = (
                deepgram_transcribe(
                    audio_path
                )
            )

            progress.progress(
                0.25
            )

            status.info(
                "🤖 မြန်မာလို ဘာသာပြန်နေသည်..."
            )

            translated_segments = (
                build_srt_segments(
                    get_gemini_client(
                        st.session_state.current_key_index
                    ),
                    source_segments,
                    lambda p, text: (
                        progress.progress(
                            min(p, 0.98)
                        ),
                        status.info(
                            text
                        ),
                    ),
                )
            )

            srt_text = make_srt(
                translated_segments
            )

            st.session_state.srt_text = (
                srt_text
            )

            st.session_state.srt_name = (
                safe_filename(
                    srt_output_name,
                    "myanmar.srt",
                )
            )

            if not (
                st.session_state
                .srt_name
                .lower()
                .endswith(".srt")
            ):

                st.session_state.srt_name += (
                    ".srt"
                )

            progress.progress(
                1.0
            )

            status.success(
                "✅ SRT ပြီးပါပြီ — "
                f"{len(translated_segments)} "
                "subtitle lines"
            )

    except Exception as exc:

        st.error(
            "SRT ထုတ်ရာမှာ အမှားဖြစ်ပါတယ်။"
        )

        st.exception(exc)


if st.session_state.srt_text:

    st.markdown(
        '<div class="srt-title">'
        '📄 Myanmar SRT Preview'
        '</div>',
        unsafe_allow_html=True,
    )

    st.text_area(
        "SRT",
        st.session_state.srt_text,
        height=300,
        label_visibility="collapsed",
        key="srt_preview",
    )

    st.download_button(
        "⬇️ Download Myanmar SRT",
        data=(
            st.session_state
            .srt_text
            .encode("utf-8-sig")
        ),
        file_name=(
            st.session_state
            .srt_name
        ),
        mime="application/x-subrip",
        use_container_width=True,
    )


# ============================================================
# STEP 2
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '② SRT → မြန်မာ Voiceover'
    '</div>',
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

    col1, col2 = st.columns(2)

    with col1:

        selected_voice = st.selectbox(
            "🎙️ Voice",
            list(VOICES.keys()),
        )

    with col2:

        selected_style = st.selectbox(
            "🎭 Voice Style",
            list(
                VOICE_STYLES.keys()
            ),
        )

    col3, col4 = st.columns(2)

    with col3:

        selected_speed = st.slider(
            "⚡ Speed",
            0.70,
            1.30,
            1.00,
            0.05,
        )

    with col4:

        output_filename = st.text_input(
            "💾 Voiceover filename",
            value=(
                "myanmar_voiceover.m4a"
            ),
        )

    make_voice_button = (
        st.form_submit_button(
            "🗣️ Voiceover ထုတ်မယ်",
            type="primary",
            use_container_width=True,
        )
    )


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

    elif st.session_state.srt_text:

        source_srt = (
            st.session_state.srt_text
        )

    if not source_srt:

        st.error(
            "SRT ဖိုင်တင်ပါ "
            "(သို့) အဆင့် ၁ မှာ SRT အရင်ထုတ်ပါ။"
        )

        st.stop()

    try:

        segments, fixed_count = (
            parse_srt(
                source_srt
            )
        )

        if fixed_count:

            st.info(
                "⏱️ SRT timing ကို "
                "အလိုအလျောက်ပြင်ပြီးပါပြီ — "
                f"{fixed_count} ခု"
            )

        else:

            st.success(
                "⏱️ SRT timing OK — "
                f"{len(segments)} lines"
            )

        with tempfile.TemporaryDirectory() as temp_dir:

            work = Path(temp_dir)

            status = st.empty()
            progress = st.progress(
                0.0
            )

            voice_path = build_voiceover(
                segments,
                VOICES[
                    selected_voice
                ],
                selected_style,
                selected_speed,
                work,
                lambda p, text: (
                    progress.progress(
                        min(p, 1.0)
                    ),
                    status.info(
                        text
                    ),
                ),
            )

            voice_bytes = (
                voice_path.read_bytes()
            )

            filename = safe_filename(
                output_filename,
                "myanmar_voiceover.m4a",
            )

            if not filename.lower().endswith(
                ".m4a"
            ):

                filename += ".m4a"

            st.session_state.voice_bytes = (
                voice_bytes
            )

            st.session_state.voice_name = (
                filename
            )

            progress.progress(
                1.0
            )

            status.success(
                "✅ Voiceover ပြီးပါပြီ"
            )

    except Exception as exc:

        st.error(
            "Voiceover ထုတ်ရာမှာ "
            "အမှားဖြစ်ပါတယ်။"
        )

        st.exception(exc)


# ============================================================
# VOICE PREVIEW
# ============================================================

if st.session_state.voice_bytes:

    st.markdown(
        '<div class="srt-title">'
        '🔊 Voiceover Preview'
        '</div>',
        unsafe_allow_html=True,
    )

    st.audio(
        st.session_state.voice_bytes,
        format="audio/mp4",
    )

    st.download_button(
        "⬇️ Download Voiceover",
        data=(
            st.session_state
            .voice_bytes
        ),
        file_name=(
            st.session_state
            .voice_name
        ),
        mime="audio/mp4",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    '🎬 Myanmar Movie AI'
    '</div>',
    unsafe_allow_html=True,
)
