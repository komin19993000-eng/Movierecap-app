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

TRANSLATION_STYLES = {
    "မူရင်းအတိုင်း သဘာဝကျကျ": "original",
    "Movie Recap Style": "recap",
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
    rows,
    translation_style="original",
    detected_language="auto",
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

    if translation_style == "recap":

        style_rules = """
TRANSLATION STYLE:
Movie Recap Style.

- Translate into natural Burmese movie-recap narration.
- Make the Burmese sound like a professional Myanmar movie recap narrator.
- Keep the important story information from the original.
- Preserve character names, actions, emotions and important details.
- Do NOT invent events that are not present in the original.
- Do NOT remove important story information just to make it shorter.
- Make the wording smooth and engaging for narration.
- Do NOT translate word-for-word when that sounds unnatural.
- Keep each subtitle suitable for its timestamp.
"""

    else:

        style_rules = """
TRANSLATION STYLE:
Original Meaning — Natural Burmese.

- Translate the original dialogue naturally into Burmese.
- Preserve the original meaning as completely as possible.
- Preserve the original information, intention, emotion and tone.
- Do NOT summarize the dialogue.
- Do NOT turn the dialogue into a movie recap.
- Do NOT intentionally remove words or important meaning just to make it shorter.
- Do NOT add information that is not in the original.
- Do NOT translate word-for-word when that sounds unnatural.
- Use natural spoken Burmese.
- Keep the translated sentence suitable for dubbing within its timestamp.
"""

    prompt = f"""
You are a professional Myanmar movie dubbing translator.

The source video language was automatically detected as:
{detected_language}

Translate every dialogue into natural spoken Burmese.

{style_rules}

IMPORTANT:
The Burmese sentence will be spoken by TTS.
Therefore it MUST be natural and easy to speak.

General rules:
- Preserve names and important proper nouns.
- Preserve emotion and intention.
- Do NOT add explanations.
- Do NOT add quotation marks unless required by meaning.
- Do NOT invent information.
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

    # Initial request + up to 5 retries.
    max_attempts = max(
        total_keys * 3,
        6,
    )

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

                # IMPORTANT:
                # Gemini translation ကို ဒီနေရာမှာ
                # character limit နဲ့ ထပ်မဖြတ်တော့ပါ။

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

            # ====================================================
            # 503 / HIGH DEMAND RETRY
            # ====================================================

            if (
                "503" in low
                or "high demand" in low
                or "unavailable" in low
                or "overloaded" in low
            ):

                attempts += 1

                if attempts < max_attempts:

                    time.sleep(
                        [
                            8,
                            16,
                            32,
                            60,
                            60,
                        ][
                            min(
                                attempts - 1,
                                4,
                            )
                        ]
                    )

                    continue

                break

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

                detected_language = "auto"

                if channels:

                    channel_language = (
                        channels[0].get(
                            "detected_language",
                            "",
                        )
                    )

                    if channel_language:
                        detected_language = (
                            str(
                                channel_language
                            )
                        )

                    alternatives = (
                        channels[0].get(
                            "alternatives",
                            [],
                        )
                        or []
                    )

                    if alternatives:

                        alternative_language = (
                            alternatives[0].get(
                                "detected_language",
                                "",
                            )
                        )

                        if alternative_language:
                            detected_language = (
                                str(
                                    alternative_language
                                )
                            )

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
                        return (
                            segments,
                            detected_language,
                        )

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
                    return (
                        fallback,
                        detected_language,
                    )

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
    translation_style="original",
    detected_language="auto",
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
                translation_style,
                detected_language,
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
        float(factor),
    )

    filters = []

    while factor > 2.0:

        filters.append(
            "atempo=2.0"
        )

        factor /= 2.0

    while factor < 0.5:

        filters.append(
            "atempo=0.5"
        )

        factor /= 0.5

    filters.append(
        f"atempo={factor:.6f}"
    )

    return ",".join(
        filters
    )


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

        # ====================================================
        # IMPORTANT FIX
        # ====================================================

        factor = max(
            required_factor,
            user_speed,
        )

    audio_filter = (
        atempo_chain(factor)
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

    # ========================================================
    # FINAL SRT TIMELINE
    # ========================================================

    timeline_end = max(
        float(item["end"])
        for item in segments
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
            "-t",
            f"{timeline_end:.3f}",
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

    selected_translation_style = st.selectbox(
        "🎬 ဘာသာပြန်ပုံစံ",
        list(
            TRANSLATION_STYLES.keys()
        ),
        index=0,
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

            (
                source_segments,
                detected_language,
            ) = deepgram_transcribe(
                audio_path
            )

            progress.progress(
                0.25
            )

            status.info(
                "🤖 "
                f"မူရင်းဘာသာစကား: "
                f"{detected_language} — "
                "မြန်မာလို ဘာသာပြန်နေသည်..."
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
                    TRANSLATION_STYLES[
                        selected_translation_style
                    ],
                    detected_language,
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
        value=st.session_state.srt_text,
        height=300,
        label_visibility="collapsed",
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
# ============================================================
# STEP 3 — VISUAL VIDEO EDITOR
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '③ 🎬 Visual Video Editor'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="
    padding:18px;
    margin-bottom:18px;
    border-radius:18px;
    border:3px solid #000000;
    background:
        linear-gradient(
            135deg,
            rgba(0,229,255,.20),
            rgba(123,60,255,.18),
            rgba(255,41,156,.18)
        );
    box-shadow:0 6px 0 #000000;
">
    <b>🎥 Video ကိုတင်ပြီး Preview Setting တွေပြောင်းကြည့်နိုင်ပါတယ်။</b><br>
    Caption, Hook, Watermark, Blur, Flip, Ratio, Background
    စတာတွေကို Editor တစ်နေရာထဲမှာ ပြင်နိုင်ပါတယ်။
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# EDITOR SESSION STATE
# ============================================================

if "editor_preview_bytes" not in st.session_state:
    st.session_state.editor_preview_bytes = None

if "editor_preview_name" not in st.session_state:
    st.session_state.editor_preview_name = "editor_preview.mp4"

if "editor_export_bytes" not in st.session_state:
    st.session_state.editor_export_bytes = None

if "editor_export_name" not in st.session_state:
    st.session_state.editor_export_name = "myanmar_edited.mp4"

if "editor_video_name" not in st.session_state:
    st.session_state.editor_video_name = ""

if "editor_srt_cache" not in st.session_state:
    st.session_state.editor_srt_cache = ""


# ============================================================
# EDITOR HELPERS
# ============================================================

def editor_run_cmd(
    args,
    timeout=1800,
):
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
    )


def editor_escape_drawtext(
    text: str,
) -> str:
    """
    FFmpeg drawtext အတွက် special characters escape လုပ်ခြင်း။
    """
    value = str(text or "")

    value = value.replace(
        "\\",
        r"\\",
    )

    value = value.replace(
        ":",
        r"\:",
    )

    value = value.replace(
        "'",
        r"\'",
    )

    value = value.replace(
        "%",
        r"\%",
    )

    value = value.replace(
        "[",
        r"\[",
    )

    value = value.replace(
        "]",
        r"\]",
    )

    value = value.replace(
        ",",
        r"\,",
    )

    return value


def editor_parse_color(
    value: str,
) -> str:
    """
    FFmpeg color value ကို safe format ပြောင်းပေးခြင်း။
    """
    value = str(value or "").strip()

    if not value:
        return "white"

    if value.startswith("#"):
        return "0x" + value[1:]

    return value


def editor_hex_to_rgb(
    value: str,
):
    value = str(value or "").strip()

    if value.startswith("#"):
        value = value[1:]

    if len(value) != 6:
        return 255, 255, 255

    try:
        return (
            int(value[0:2], 16),
            int(value[2:4], 16),
            int(value[4:6], 16),
        )
    except Exception:
        return 255, 255, 255


def editor_get_font_path():
    """
    Server environment ထဲမှာ ရှိနိုင်တဲ့ font ကိုရှာမယ်။
    Myanmar font မတွေ့ရင် DejaVu Sans ကို fallback သုံးမယ်။
    """

    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMyanmar-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for path in candidates:

        if Path(path).exists():
            return path

    return ""


def editor_get_video_info(
    video_path: Path,
):
    result = editor_run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(video_path),
        ],
        timeout=120,
    )

    stderr = result.stderr or ""

    duration_match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        stderr,
    )

    width_match = re.search(
        r"Video:.*?\s(\d{2,5})x(\d{2,5})",
        stderr,
    )

    duration = 0.0
    width = 0
    height = 0

    if duration_match:

        duration = (
            int(duration_match.group(1))
            * 3600
            + int(duration_match.group(2))
            * 60
            + float(duration_match.group(3))
        )

    if width_match:

        width = int(
            width_match.group(1)
        )

        height = int(
            width_match.group(2)
        )

    return duration, width, height


def editor_load_srt(
    srt_text: str,
):
    if not srt_text.strip():
        return []

    try:

        entries, _ = parse_srt(
            srt_text
        )

        return entries

    except Exception:

        return []


def editor_find_active_caption(
    entries,
    timestamp: float,
):
    for item in entries:

        start = float(
            item["start"]
        )

        end = float(
            item["end"]
        )

        if (
            start <= timestamp <= end
        ):

            return clean_text(
                item.get(
                    "burmese",
                    "",
                )
            )

    return ""


def editor_build_preview(
    video_path: Path,
    output_path: Path,
    preview_time: float,
    ratio: str,
    quality: str,
    flip_video: bool,
    brightness: float,
    contrast: float,
    saturation: float,
    blur_enabled: bool,
    blur_strength: int,
    caption_enabled: bool,
    caption_text: str,
    caption_color: str,
    caption_bg: str,
    caption_bg_opacity: float,
    caption_size: int,
    caption_position: str,
    hook_enabled: bool,
    hook_text: str,
    hook_color: str,
    hook_size: int,
    hook_position: str,
    watermark_enabled: bool,
    watermark_text: str,
    watermark_opacity: float,
    logo_path: Path | None,
    background: str,
):
    """
    Low-resolution preview renderer.
    User setting ပြောင်းတိုင်း ခေါ်နိုင်အောင်
    preview render ကို မြန်အောင် 720p အောက်သို့ scale လုပ်ထားသည်။
    """

    preview_time = max(
        0.0,
        float(preview_time),
    )

    filters = []

    # --------------------------------------------------------
    # SOURCE SCALE
    # --------------------------------------------------------

    if ratio == "TikTok / Reels — 9:16":

        filters.append(
            "scale=720:-2:force_original_aspect_ratio=decrease"
        )

        filters.append(
            "pad=720:1280:(ow-iw)/2:(oh-ih)/2:"
            f"color={editor_parse_color(background)}"
        )

    elif ratio == "YouTube — 16:9":

        filters.append(
            "scale=1280:-2:force_original_aspect_ratio=decrease"
        )

        filters.append(
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2:"
            f"color={editor_parse_color(background)}"
        )

    else:

        filters.append(
            "scale=720:720:force_original_aspect_ratio=decrease"
        )

        filters.append(
            "pad=720:720:(ow-iw)/2:(oh-ih)/2:"
            f"color={editor_parse_color(background)}"
        )

    # --------------------------------------------------------
    # FLIP
    # --------------------------------------------------------

    if flip_video:

        filters.append(
            "hflip"
        )

    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    color_filter = (
        f"eq="
        f"brightness={float(brightness):.3f}:"
        f"contrast={float(contrast):.3f}:"
        f"saturation={float(saturation):.3f}"
    )

    filters.append(
        color_filter
    )

    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    if blur_enabled:

        blur_value = max(
            1,
            min(
                int(blur_strength),
                40,
            ),
        )

        filters.append(
            f"boxblur={blur_value}:1"
        )

    # --------------------------------------------------------
    # CAPTION
    # --------------------------------------------------------

    font_path = editor_get_font_path()

    if caption_enabled and caption_text:

        escaped_caption = (
            editor_escape_drawtext(
                caption_text
            )
        )

        caption_color_ff = (
            editor_parse_color(
                caption_color
            )
        )

        bg_color = editor_parse_color(
            caption_bg
        )

        bg_rgb = editor_hex_to_rgb(
            caption_bg
        )

        alpha = max(
            0.0,
            min(
                1.0,
                float(
                    caption_bg_opacity
                ),
            ),
        )

        bg_alpha = int(
            round(
                alpha * 255
            )
        )

        if caption_position == "အပေါ်":

            y_expr = "70"

        elif caption_position == "အလယ်":

            y_expr = "(h-text_h)/2"

        else:

            y_expr = "h-text_h-70"

        drawtext_parts = [
            f"text='{escaped_caption}'",
            f"fontcolor={caption_color_ff}",
            f"fontsize={int(caption_size)}",
            "x=(w-text_w)/2",
            f"y={y_expr}",
            "borderw=2",
            "bordercolor=black",
        ]

        if font_path:

            drawtext_parts.append(
                "fontfile="
                + font_path
            )

        # Semi-transparent background box
        if bg_alpha > 0:

            drawtext_parts.extend(
                [
                    f"box=1",
                    f"boxcolor="
                    f"{bg_color}@"
                    f"{alpha:.2f}",
                    "boxborderw=12",
                ]
            )

        filters.append(
            "drawtext="
            + ":".join(
                drawtext_parts
            )
        )

    # --------------------------------------------------------
    # HOOK
    # --------------------------------------------------------

    if hook_enabled and hook_text:

        escaped_hook = (
            editor_escape_drawtext(
                hook_text
            )
        )

        hook_color_ff = (
            editor_parse_color(
                hook_color
            )
        )

        if hook_position == "အပေါ်":

            hook_y = "45"

        elif hook_position == "အလယ်":

            hook_y = "(h-text_h)/2"

        else:

            hook_y = "h-text_h-120"

        hook_parts = [
            f"text='{escaped_hook}'",
            f"fontcolor={hook_color_ff}",
            f"fontsize={int(hook_size)}",
            "x=(w-text_w)/2",
            f"y={hook_y}",
            "borderw=3",
            "bordercolor=black",
            "box=1",
            "boxcolor=black@0.35",
            "boxborderw=10",
        ]

        if font_path:

            hook_parts.append(
                "fontfile="
                + font_path
            )

        filters.append(
            "drawtext="
            + ":".join(
                hook_parts
            )
        )

    # --------------------------------------------------------
    # WATERMARK
    # --------------------------------------------------------

    if watermark_enabled and watermark_text:

        escaped_watermark = (
            editor_escape_drawtext(
                watermark_text
            )
        )

        watermark_alpha = max(
            0.0,
            min(
                1.0,
                float(
                    watermark_opacity
                ),
            ),
        )

        watermark_parts = [
            f"text='{escaped_watermark}'",
            f"fontcolor=white@"
            f"{watermark_alpha:.2f}",
            "fontsize=22",
            "x=w-text_w-25",
            "y=h-text_h-25",
            "borderw=2",
            "bordercolor=black@0.55",
        ]

        if font_path:

            watermark_parts.append(
                "fontfile="
                + font_path
            )

        filters.append(
            "drawtext="
            + ":".join(
                watermark_parts
            )
        )

    # --------------------------------------------------------
    # VIDEO PREVIEW OUTPUT
    # --------------------------------------------------------

    vf = ",".join(
        filters
    )

    command = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{preview_time:.3f}",
        "-i",
        str(video_path),
        "-t",
        "8",
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = editor_run_cmd(
        command,
        timeout=300,
    )

    if (
        result.returncode != 0
        or not output_path.exists()
        or output_path.stat().st_size < 1000
    ):

        raise RuntimeError(
            "Preview render မအောင်မြင်ပါ။\n"
            + (
                result.stderr
                or ""
            )
        )


def editor_export_video(
    video_path: Path,
    output_path: Path,
    ratio: str,
    quality: str,
    flip_video: bool,
    brightness: float,
    contrast: float,
    saturation: float,
    blur_enabled: bool,
    blur_strength: int,
    caption_enabled: bool,
    caption_text: str,
    caption_color: str,
    caption_bg: str,
    caption_bg_opacity: float,
    caption_size: int,
    caption_position: str,
    hook_enabled: bool,
    hook_text: str,
    hook_color: str,
    hook_size: int,
    hook_position: str,
    watermark_enabled: bool,
    watermark_text: str,
    watermark_opacity: float,
    background: str,
):
    """
    Final video export.
    """

    filters = []

    # --------------------------------------------------------
    # RATIO
    # --------------------------------------------------------

    if ratio == "TikTok / Reels — 9:16":

        filters.append(
            "scale=720:1280:force_original_aspect_ratio=decrease"
        )

        filters.append(
            "pad=720:1280:(ow-iw)/2:(oh-ih)/2:"
            f"color={editor_parse_color(background)}"
        )

    elif ratio == "YouTube — 16:9":

        filters.append(
            "scale=1280:720:force_original_aspect_ratio=decrease"
        )

        filters.append(
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2:"
            f"color={editor_parse_color(background)}"
        )

    else:

        filters.append(
            "scale=720:720:force_original_aspect_ratio=decrease"
        )

        filters.append(
            "pad=720:720:(ow-iw)/2:(oh-ih)/2:"
            f"color={editor_parse_color(background)}"
        )

    # --------------------------------------------------------
    # FLIP
    # --------------------------------------------------------

    if flip_video:

        filters.append(
            "hflip"
        )

    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    filters.append(
        f"eq="
        f"brightness={float(brightness):.3f}:"
        f"contrast={float(contrast):.3f}:"
        f"saturation={float(saturation):.3f}"
    )

    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    if blur_enabled:

        blur_value = max(
            1,
            min(
                int(blur_strength),
                40,
            ),
        )

        filters.append(
            f"boxblur={blur_value}:1"
        )

    font_path = editor_get_font_path()

    # --------------------------------------------------------
    # CAPTION
    # --------------------------------------------------------

    if caption_enabled and caption_text:

        escaped_caption = (
            editor_escape_drawtext(
                caption_text
            )
        )

        caption_color_ff = (
            editor_parse_color(
                caption_color
            )
        )

        bg_color = editor_parse_color(
            caption_bg
        )

        alpha = max(
            0.0,
            min(
                1.0,
                float(
                    caption_bg_opacity
                ),
            ),
        )

        if caption_position == "အပေါ်":

            caption_y = "70"

        elif caption_position == "အလယ်":

            caption_y = "(h-text_h)/2"

        else:

            caption_y = "h-text_h-70"

        parts = [
            f"text='{escaped_caption}'",
            f"fontcolor={caption_color_ff}",
            f"fontsize={int(caption_size)}",
            "x=(w-text_w)/2",
            f"y={caption_y}",
            "borderw=2",
            "bordercolor=black",
        ]

        if font_path:

            parts.append(
                "fontfile="
                + font_path
            )

        if alpha > 0:

            parts.extend(
                [
                    "box=1",
                    f"boxcolor="
                    f"{bg_color}@"
                    f"{alpha:.2f}",
                    "boxborderw=12",
                ]
            )

        filters.append(
            "drawtext="
            + ":".join(parts)
        )

    # --------------------------------------------------------
    # HOOK
    # --------------------------------------------------------

    if hook_enabled and hook_text:

        escaped_hook = (
            editor_escape_drawtext(
                hook_text
            )
        )

        hook_color_ff = (
            editor_parse_color(
                hook_color
            )
        )

        if hook_position == "အပေါ်":

            hook_y = "45"

        elif hook_position == "အလယ်":

            hook_y = "(h-text_h)/2"

        else:

            hook_y = "h-text_h-120"

        parts = [
            f"text='{escaped_hook}'",
            f"fontcolor={hook_color_ff}",
            f"fontsize={int(hook_size)}",
            "x=(w-text_w)/2",
            f"y={hook_y}",
            "borderw=3",
            "bordercolor=black",
            "box=1",
            "boxcolor=black@0.35",
            "boxborderw=10",
        ]

        if font_path:

            parts.append(
                "fontfile="
                + font_path
            )

        filters.append(
            "drawtext="
            + ":".join(parts)
        )

    # --------------------------------------------------------
    # WATERMARK
    # --------------------------------------------------------

    if watermark_enabled and watermark_text:

        escaped_watermark = (
            editor_escape_drawtext(
                watermark_text
            )
        )

        opacity = max(
            0.0,
            min(
                1.0,
                float(
                    watermark_opacity
                ),
            ),
        )

        parts = [
            f"text='{escaped_watermark}'",
            f"fontcolor=white@"
            f"{opacity:.2f}",
            "fontsize=22",
            "x=w-text_w-25",
            "y=h-text_h-25",
            "borderw=2",
            "bordercolor=black@0.55",
        ]

        if font_path:

            parts.append(
                "fontfile="
                + font_path
            )

        filters.append(
            "drawtext="
            + ":".join(parts)
        )

    vf = ",".join(
        filters
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    if quality == "High":

        crf = "20"

    elif quality == "Medium":

        crf = "23"

    else:

        crf = "27"

    command = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        crf,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = editor_run_cmd(
        command,
        timeout=3600,
    )

    if (
        result.returncode != 0
        or not output_path.exists()
        or output_path.stat().st_size < 5000
    ):

        raise RuntimeError(
            "Edited video export မအောင်မြင်ပါ။\n"
            + (
                result.stderr
                or ""
            )
        )


# ============================================================
# EDITOR SOURCE VIDEO
# ============================================================

editor_video = st.file_uploader(
    "🎥 Editor အတွက် Video တင်ပါ",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm",
    ],
    key="visual_editor_video",
)


if editor_video:

    st.session_state.editor_video_name = (
        editor_video.name
    )

    editor_temp_dir = Path(
        tempfile.gettempdir()
    ) / "myanmar_movie_ai_editor"

    editor_temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    editor_source_path = (
        editor_temp_dir
        / safe_filename(
            editor_video.name,
            "editor_input.mp4",
        )
    )

    editor_source_path.write_bytes(
        editor_video.getbuffer()
    )

    # --------------------------------------------------------
    # OPTIONAL SRT
    # --------------------------------------------------------

    st.markdown(
        "### 📝 Caption Source"
    )

    editor_srt_source = st.text_area(
        "SRT",
        value=(
            st.session_state.srt_text
            if st.session_state.srt_text
            else ""
        ),
        height=160,
        key="editor_srt_text",
        help=(
            "အဆင့် ၁ က Myanmar SRT ရှိရင် "
            "အလိုအလျောက်ထည့်ပေးထားပါမယ်။"
        ),
    )

    editor_entries = (
        editor_load_srt(
            editor_srt_source
        )
    )

    # --------------------------------------------------------
    # VIDEO INFO
    # --------------------------------------------------------

    try:

        (
            editor_duration,
            editor_width,
            editor_height,
        ) = editor_get_video_info(
            editor_source_path
        )

    except Exception:

        editor_duration = 0.0
        editor_width = 0
        editor_height = 0

    if editor_duration > 0:

        st.caption(
            f"🎞️ {editor_width}×{editor_height}  "
            f"• {editor_duration:.1f} sec"
        )

    # --------------------------------------------------------
    # EDITOR LAYOUT
    # --------------------------------------------------------

    editor_left, editor_right = (
        st.columns(
            [1.25, 1],
            gap="large",
        )
    )

    # ========================================================
    # LEFT — VIDEO PREVIEW
    # ========================================================

    with editor_left:

        st.markdown(
            "### 🎥 Live Preview"
        )

        if editor_duration > 0:

            preview_time = st.slider(
                "Preview Position",
                0.0,
                float(
                    max(
                        0.1,
                        editor_duration - 0.1,
                    )
                ),
                0.0,
                0.5,
                key="editor_preview_time",
            )

        else:

            preview_time = 0.0

        st.video(
            str(
                editor_source_path
            ),
            start_time=int(
                preview_time
            ),
        )

        st.caption(
            "Setting ပြောင်းပြီး Preview ကို "
            "ကြည့်နိုင်ပါတယ်။"
        )

    # ========================================================
    # RIGHT — CONTROLS
    # ========================================================

    with editor_right:

        st.markdown(
            "### ⚙️ Editor Controls"
        )

        # ----------------------------------------------------
        # VIDEO FORMAT
        # ----------------------------------------------------

        st.markdown(
            "#### 📐 Video Size"
        )

        editor_ratio = st.selectbox(
            "Platform Ratio",
            [
                "TikTok / Reels — 9:16",
                "YouTube — 16:9",
                "Square — 1:1",
            ],
            key="editor_ratio",
        )

        editor_quality = st.selectbox(
            "Export Quality",
            [
                "High",
                "Medium",
                "Fast",
            ],
            index=1,
            key="editor_quality",
        )

        editor_background = st.color_picker(
            "🖼️ Background",
            "#000000",
            key="editor_background",
        )

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        st.markdown(
            "#### 🔄 Transform"
        )

        editor_flip = st.checkbox(
            "🔄 Flip Video",
            value=False,
            key="editor_flip",
        )

        # ----------------------------------------------------
        # COLOR
        # ----------------------------------------------------

        st.markdown(
            "#### 🎨 Color"
        )

        editor_brightness = st.slider(
            "Brightness",
            -1.0,
            1.0,
            0.0,
            0.05,
            key="editor_brightness",
        )

        editor_contrast = st.slider(
            "Contrast",
            0.5,
            2.0,
            1.0,
            0.05,
            key="editor_contrast",
        )

        editor_saturation = st.slider(
            "Saturation",
            0.0,
            2.0,
            1.0,
            0.05,
            key="editor_saturation",
        )

        # ----------------------------------------------------
        # BLUR
        # ----------------------------------------------------

        st.markdown(
            "#### 🫥 Blur / Mask"
        )

        editor_blur = st.checkbox(
            "🫥 Blur Video",
            value=False,
            key="editor_blur",
        )

        editor_blur_strength = st.slider(
            "Blur Strength",
            1,
            30,
            8,
            1,
            disabled=not editor_blur,
            key="editor_blur_strength",
        )

        # ----------------------------------------------------
        # CAPTION
        # ----------------------------------------------------

        st.markdown(
            "#### 📝 Caption"
        )

        editor_caption_enabled = st.checkbox(
            "📝 Show Caption",
            value=bool(
                editor_entries
            ),
            key="editor_caption_enabled",
        )

        editor_caption_mode = st.radio(
            "Caption Source",
            [
                "SRT အလိုအလျောက်",
                "Manual",
            ],
            horizontal=True,
            key="editor_caption_mode",
        )

        if (
            editor_caption_mode
            == "SRT အလိုအလျောက်"
        ):

            auto_caption = (
                editor_find_active_caption(
                    editor_entries,
                    preview_time,
                )
            )

            editor_caption_text = (
                auto_caption
            )

        else:

            editor_caption_text = st.text_input(
                "Caption Text",
                value="",
                key="editor_manual_caption",
            )

        editor_caption_color = st.color_picker(
            "Caption Color",
            "#FFFFFF",
            key="editor_caption_color",
        )

        editor_caption_bg = st.color_picker(
            "Caption Background",
            "#000000",
            key="editor_caption_bg",
        )

        editor_caption_bg_opacity = (
            st.slider(
                "Caption Background Transparency",
                0.0,
                1.0,
                0.55,
                0.05,
                key="editor_caption_bg_opacity",
            )
        )

        editor_caption_size = st.slider(
            "Caption Font Size",
            20,
            80,
            42,
            2,
            key="editor_caption_size",
        )

        editor_caption_position = st.selectbox(
            "Caption Position",
            [
                "အပေါ်",
                "အလယ်",
                "အောက်",
            ],
            index=2,
            key="editor_caption_position",
        )

        # ----------------------------------------------------
        # HOOK
        # ----------------------------------------------------

        st.markdown(
            "#### 🎯 Hook"
        )

        editor_hook_enabled = st.checkbox(
            "🎯 Show Hook",
            value=False,
            key="editor_hook_enabled",
        )

        editor_hook_text = st.text_input(
            "Hook Text",
            value="ဒီဇာတ်ကားမှာ ဘာဖြစ်မလဲ?",
            disabled=not editor_hook_enabled,
            key="editor_hook_text",
        )

        editor_hook_color = st.color_picker(
            "Hook Color",
            "#FFFFFF",
            disabled=not editor_hook_enabled,
            key="editor_hook_color",
        )

        editor_hook_size = st.slider(
            "Hook Size",
            25,
            100,
            52,
            2,
            disabled=not editor_hook_enabled,
            key="editor_hook_size",
        )

        editor_hook_position = st.selectbox(
            "Hook Position",
            [
                "အပေါ်",
                "အလယ်",
                "အောက်",
            ],
            index=0,
            disabled=not editor_hook_enabled,
            key="editor_hook_position",
        )

        # ----------------------------------------------------
        # WATERMARK
        # ----------------------------------------------------

        st.markdown(
            "#### 💧 Watermark"
        )

        editor_watermark_enabled = st.checkbox(
            "💧 Show Watermark",
            value=False,
            key="editor_watermark_enabled",
        )

        editor_watermark_text = st.text_input(
            "Watermark Text",
            value="Myanmar Movie AI",
            disabled=not editor_watermark_enabled,
            key="editor_watermark_text",
        )

        editor_watermark_opacity = st.slider(
            "Watermark Opacity",
            0.10,
            1.0,
            0.55,
            0.05,
            disabled=not editor_watermark_enabled,
            key="editor_watermark_opacity",
        )

    # ========================================================
    # PREVIEW RENDER BUTTON
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### 👀 Edited Preview"
    )

    preview_col1, preview_col2 = st.columns(
        2
    )

    with preview_col1:

        render_preview_button = st.button(
            "👀 Apply & Preview",
            type="primary",
            use_container_width=True,
            key="editor_render_preview",
        )

    with preview_col2:

        export_video_button = st.button(
            "🎬 Export Edited Video",
            type="primary",
            use_container_width=True,
            key="editor_export_video",
        )

    # ========================================================
    # RENDER PREVIEW
    # ========================================================

    if render_preview_button:

        try:

            preview_output = (
                editor_temp_dir
                / "editor_preview.mp4"
            )

            with st.spinner(
                "🎬 Preview ပြင်ဆင်နေသည်..."
            ):

                editor_build_preview(
                    editor_source_path,
                    preview_output,
                    preview_time,
                    editor_ratio,
                    editor_quality,
                    editor_flip,
                    editor_brightness,
                    editor_contrast,
                    editor_saturation,
                    editor_blur,
                    editor_blur_strength,
                    editor_caption_enabled,
                    editor_caption_text,
                    editor_caption_color,
                    editor_caption_bg,
                    editor_caption_bg_opacity,
                    editor_caption_size,
                    editor_caption_position,
                    editor_hook_enabled,
                    editor_hook_text,
                    editor_hook_color,
                    editor_hook_size,
                    editor_hook_position,
                    editor_watermark_enabled,
                    editor_watermark_text,
                    editor_watermark_opacity,
                    None,
                    editor_background,
                )

            st.session_state.editor_preview_bytes = (
                preview_output.read_bytes()
            )

            st.session_state.editor_preview_name = (
                "editor_preview.mp4"
            )

            st.success(
                "✅ Edited Preview ပြီးပါပြီ"
            )

        except Exception as exc:

            st.error(
                "Preview ထုတ်ရာမှာ အမှားဖြစ်ပါတယ်။"
            )

            st.exception(exc)

    # ========================================================
    # SHOW RENDERED PREVIEW
    # ========================================================

    if st.session_state.editor_preview_bytes:

        st.video(
            st.session_state.editor_preview_bytes
        )

    # ========================================================
    # FINAL EXPORT
    # ========================================================

    if export_video_button:

        try:

            export_output = (
                editor_temp_dir
                / "myanmar_edited.mp4"
            )

            with st.spinner(
                "🎬 Final Edited Video ထုတ်နေသည်..."
            ):

                editor_export_video(
                    editor_source_path,
                    export_output,
                    editor_ratio,
                    editor_quality,
                    editor_flip,
                    editor_brightness,
                    editor_contrast,
                    editor_saturation,
                    editor_blur,
                    editor_blur_strength,
                    editor_caption_enabled,
                    editor_caption_text,
                    editor_caption_color,
                    editor_caption_bg,
                    editor_caption_bg_opacity,
                    editor_caption_size,
                    editor_caption_position,
                    editor_hook_enabled,
                    editor_hook_text,
                    editor_hook_color,
                    editor_hook_size,
                    editor_hook_position,
                    editor_watermark_enabled,
                    editor_watermark_text,
                    editor_watermark_opacity,
                    editor_background,
                )

            st.session_state.editor_export_bytes = (
                export_output.read_bytes()
            )

            st.session_state.editor_export_name = (
                safe_filename(
                    Path(
                        editor_video.name
                    ).stem
                    + "_edited.mp4",
                    "myanmar_edited.mp4",
                )
            )

            st.success(
                "✅ Edited Video ပြီးပါပြီ"
            )

        except Exception as exc:

            st.error(
                "Edited Video ထုတ်ရာမှာ "
                "အမှားဖြစ်ပါတယ်။"
            )

            st.exception(exc)

    # ========================================================
    # FINAL VIDEO
    # ========================================================

    if st.session_state.editor_export_bytes:

        st.markdown(
            "### 🎬 Final Edited Video"
        )

        st.video(
            st.session_state.editor_export_bytes
        )

        st.download_button(
            "⬇️ Download Edited Video",
            data=(
                st.session_state
                .editor_export_bytes
            ),
            file_name=(
                st.session_state
                .editor_export_name
            ),
            mime="video/mp4",
            use_container_width=True,
            key="download_editor_video",
        )

else:

    st.info(
        "🎥 Visual Editor သုံးရန် Video တစ်ခုတင်ပါ။"
    )
