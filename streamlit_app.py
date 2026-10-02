import asyncio
import json
import os
import random
import re
import shutil
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
# VOICEOVER TIMING POLICY
# ============================================================
# Speech sped up beyond MAX_INTELLIGIBLE_SPEEDUP becomes
# unintelligible ("chipmunk" audio). The fitter never exceeds
# this cap. Over-long narration is rewritten shorter once via
# AI (proper paraphrase, not truncation); any remainder is
# allowed to overflow slightly into the following pause.
MAX_INTELLIGIBLE_SPEEDUP = 1.35

# Rough Burmese TTS rate (chars/sec), used only to budget the
# one-shot AI rewrite. The real gate is re-measuring the TTS
# file afterwards, so an imperfect estimate is safe.
REWRITE_TARGET_CPS = 11.0

# Persistent work dir for the Edit step (Step 1/2 outputs that
# the editor reuses). Lives for the process lifetime.
EDIT_WORK_DIR = Path(
    tempfile.mkdtemp(
        prefix="movierecap_edit_"
    )
)


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


def save_uploaded_file(uploaded_file, dest_path):
    """Write a Streamlit UploadedFile to disk in 1MB chunks.

    Avoids uploaded_file.getbuffer(), which duplicates the whole
    file in RAM — matters on the 1GB Streamlit Cloud free tier
    when uploads approach the size limit.
    """

    dest_path = Path(dest_path)

    uploaded_file.seek(0)

    with open(dest_path, "wb") as out:
        shutil.copyfileobj(
            uploaded_file,
            out,
            length=1024 * 1024,
        )

    uploaded_file.seek(0)

    return dest_path


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


def rewrite_shorter_burmese(
    client,
    text: str,
    max_chars: int,
):
    """Ask Gemini to paraphrase Burmese narration more concisely.

    Unlike hard character truncation (which cuts mid-sentence and
    destroys meaning), this rewrites with complete sentences.
    Returns the shorter text, or None when rewriting failed or
    did not shorten anything.
    """

    text = clean_text(text)

    if not text:
        return None

    prompt = (
        "You are a professional Myanmar dubbing scriptwriter.\n"
        "Rewrite the Burmese narration below so it is SHORTER "
        "and fits within about "
        f"{max(8, int(max_chars))} characters.\n"
        "Rules:\n"
        "- Use complete sentences only. NEVER cut a sentence off.\n"
        "- Keep the core meaning, character names and emotion.\n"
        "- Drop filler words and secondary detail.\n"
        "- Natural spoken Burmese.\n"
        "- Do NOT add quotation marks.\n"
        "- Return ONLY the rewritten Burmese text, no explanation.\n"
        "\n"
        f"Original ({len(text)} characters):\n"
        f"{text}"
    )

    try:

        response = (
            client.models.generate_content(
                model=get_gemini_model(),
                contents=prompt,
            )
        )

        new_text = clean_text(
            getattr(
                response,
                "text",
                "",
            )
        ).strip(
            "\"\u201c\u201d'"
        ).strip()

        if (
            new_text
            and len(new_text)
            < len(text)
        ):
            return new_text

    except Exception:
        pass

    return None


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
        # INTELLIGIBILITY CAP.
        # Never speed speech beyond MAX_INTELLIGIBLE_SPEEDUP.
        # If the narration is longer than the slot even at that
        # speed, the caller may first try an AI rewrite; the
        # remainder is allowed to overflow slightly into the
        # following pause instead of becoming chipmunk audio.
        # ====================================================

        factor = min(
            max(
                required_factor,
                user_speed,
            ),
            MAX_INTELLIGIBLE_SPEEDUP,
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
    gemini_client=None,
):

    clips = []
    total = len(segments)

    # Lines that were rewritten shorter by AI, and lines that
    # still overflow their slot even at the intelligibility cap.
    rewritten_count = 0
    overflow_lines = []

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

        # If the narration cannot fit the slot at an
        # intelligible speed, try ONE proper AI rewrite
        # (complete sentences, not truncation) before
        # accepting a slight overflow into the next pause.

        raw_duration = ffprobe_duration(
            raw
        )

        if (
            gemini_client is not None
            and raw_duration
            / max(slot, 0.20)
            > MAX_INTELLIGIBLE_SPEEDUP
        ):

            progress_callback(
                0.05
                + 0.75
                * (index / total),
                (
                    f"Line {index}/{total} "
                    "too long — rewriting shorter..."
                ),
            )

            target_chars = max(
                8,
                int(
                    slot
                    * MAX_INTELLIGIBLE_SPEEDUP
                    * REWRITE_TARGET_CPS
                ),
            )

            try:

                shorter = (
                    rewrite_shorter_burmese(
                        gemini_client,
                        text,
                        target_chars,
                    )
                )

            except Exception:

                shorter = None

            if shorter:

                retry_raw = (
                    work_dir
                    / f"tts_{index:04d}_r.mp3"
                )

                try:

                    make_tts(
                        shorter,
                        voice,
                        style,
                        retry_raw,
                    )

                    retry_raw.replace(
                        raw
                    )

                    text = shorter

                    item["burmese"] = (
                        shorter
                    )

                    rewritten_count += 1

                    raw_duration = (
                        ffprobe_duration(
                            raw
                        )
                    )

                except Exception:

                    if retry_raw.exists():
                        retry_raw.unlink()

        fit_tts_to_slot(
            raw,
            fitted,
            slot,
            speed,
        )

        overflow = (
            raw_duration
            / MAX_INTELLIGIBLE_SPEEDUP
            - slot
        )

        if overflow > 0.05:

            overflow_lines.append(
                (
                    index,
                    round(
                        overflow,
                        2,
                    ),
                )
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

    return (
        output,
        segments,
        {
            "rewritten": rewritten_count,
            "overflow": overflow_lines,
        },
    )


# ============================================================
# EDIT & EXPORT HELPERS
# ============================================================

def probe_media(
    path: Path
) -> dict:
    """Width/height/duration + stream presence via ffmpeg -i."""

    info = {
        "width": 0,
        "height": 0,
        "duration": 0.0,
        "has_audio": False,
        "has_video": False,
        "fps": 30.0,
    }

    result = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(path),
        ],
        timeout=60,
    )

    err = result.stderr or ""

    match = re.search(
        r"Duration:\s*(\d+):(\d+):([\d.]+)",
        err,
    )

    if match:
        info["duration"] = (
            int(match.group(1)) * 3600
            + int(match.group(2)) * 60
            + float(match.group(3))
        )

    video_match = re.search(
        r"Stream #\d+:\d+[^:]*: Video:[^\n]*?"
        r"(\d{2,5})x(\d{2,5})",
        err,
    )

    if video_match:
        info["width"] = int(
            video_match.group(1)
        )
        info["height"] = int(
            video_match.group(2)
        )
        info["has_video"] = True

        fps_match = re.search(
            r"(\d+(?:\.\d+)?)\s*fps",
            video_match.group(0),
        )

        if fps_match:

            try:
                info["fps"] = float(
                    fps_match.group(1)
                )
            except ValueError:
                pass

    if re.search(
        r"Stream #\d+:\d+[^:]*: Audio:",
        err,
    ):
        info["has_audio"] = True

    return info


def find_myanmar_font() -> str:
    """A font family with Myanmar glyphs, or '' when none found."""

    try:

        result = subprocess.run(
            [
                "fc-list",
                ":lang=my",
                "family",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )

        for line in (
            result.stdout or ""
        ).splitlines():

            family = line.split(
                ","
            )[0].strip()

            if family:
                return family

    except Exception:
        pass

    return ""


def mask_filter_chain(
    start_label: str,
    masks: list,
    blur_radius: int,
    frame_w: int = 0,
    frame_h: int = 0,
):
    """Cover regions (hardcoded subtitles) with blur or black box."""

    parts = []
    current = start_label

    for i, mask in enumerate(masks):

        x = max(0, int(mask["x"]))
        y = max(0, int(mask["y"]))
        w = max(2, int(mask["w"]))
        h = max(2, int(mask["h"]))

        if frame_w > 0 and frame_h > 0:

            # Defensive: a mask must never exceed the
            # frame, no matter how upstream rounding
            # moved things. Prevents "Invalid too big
            # ... size" crop crashes.

            x = min(x, frame_w - 2)
            y = min(y, frame_h - 2)
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)
            w = max(2, w)
            h = max(2, h)

        if mask.get("mode") == "blur":

            # boxblur rejects radius >= half the frame's
            # smallest chroma-plane dimension, so clamp to
            # the mask region (small strips on small videos
            # would otherwise crash the render).

            radius = max(
                2,
                min(
                    int(blur_radius),
                    (min(w, h) - 1) // 4,
                ),
            )

            parts.append(
                f"{current}split=2"
                f"[vin{i}][vcp{i}]"
            )

            parts.append(
                f"[vin{i}]"
                f"crop={w}:{h}:{x}:{y},"
                f"boxblur=luma_radius={radius}:"
                f"luma_power=3"
                f"[mk{i}]"
            )

            parts.append(
                f"[vcp{i}][mk{i}]"
                f"overlay={x}:{y}"
                f"[vmsk{i}]"
            )

        else:

            parts.append(
                f"{current}"
                f"drawbox=x={x}:y={y}:"
                f"w={w}:h={h}:"
                f"color=black:t=fill"
                f"[vmsk{i}]"
            )

        current = f"[vmsk{i}]"

    return (
        ";".join(parts),
        current,
    )


def transform_filter_chain(
    start_label: str,
    flip: bool,
    zoom_mode: str,
    zoom_amount: float,
    eq_preset: str,
    src_w: int,
    src_h: int,
    src_fps: float,
    zoom_cx: float = 50.0,
    zoom_cy: float = 50.0,
):
    """Copyright-evasion transforms: flip -> zoom -> color filter.

    Output frame keeps the source WxH so downstream
    masks / burned subs / ratio math stays valid.
    """

    parts = []
    current = start_label

    if flip:

        parts.append(
            f"{current}hflip[vflip]"
        )

        current = "[vflip]"

    if zoom_mode == "static":

        z = max(1.0, float(zoom_amount))

        # Integer-exact math: the old float chain
        # (scale=iw*z then crop=iw/z) could round-trip
        # 1px short (e.g. 1024x576 at 1.15x became
        # 1023x575), which then broke pixel-exact
        # downstream masks with "Invalid too big ...
        # size" crop errors. Everything here is ints,
        # so the output is exactly ew x eh.

        ew = (max(2, int(src_w)) // 2) * 2
        eh = (max(2, int(src_h)) // 2) * 2
        sw = max(ew + 2, int(round(ew * z)))
        sh = max(eh + 2, int(round(eh * z)))

        # Free zoom center: 0/0 = top-left,
        # 50/50 = center, 100/100 = bottom-right.

        cx = (
            min(100.0, max(0.0, float(zoom_cx)))
            / 100.0
        )
        cy = (
            min(100.0, max(0.0, float(zoom_cy)))
            / 100.0
        )

        ox = int(round((sw - ew) * cx))
        oy = int(round((sh - eh) * cy))

        parts.append(
            f"{current}"
            f"scale={sw}:{sh},"
            f"crop={ew}:{eh}:{ox}:{oy}"
            "[vzoom]"
        )

        current = "[vzoom]"

    elif zoom_mode == "dynamic":

        w = (max(2, int(src_w)) // 2) * 2
        h = (max(2, int(src_h)) // 2) * 2
        fps = max(1.0, float(src_fps or 30.0))

        # Slow push-in: 1.00 -> 1.12 over ~60 seconds.
        # 'in' counts input frames; fps is pinned to the
        # source so audio/subtitle timing never drifts.
        parts.append(
            f"{current}"
            "zoompan="
            "z='1.0+0.12*min(in/1800\\,1)':"
            "d=1:"
            "x='iw/2-(iw/zoom/2)':"
            "y='ih/2-(ih/zoom/2)':"
            f"s={w}x{h}:"
            f"fps={fps:.2f}"
            "[vzoom]"
        )

        current = "[vzoom]"

    eq = {
        "vivid": (
            "eq=saturation=1.30:"
            "contrast=1.06:"
            "brightness=0.015"
        ),
        "warm": (
            "eq=saturation=1.15:"
            "contrast=1.03:"
            "gamma_r=1.05:"
            "gamma_b=0.95"
        ),
        "cool": (
            "eq=saturation=1.15:"
            "contrast=1.03:"
            "gamma_r=0.95:"
            "gamma_b=1.05"
        ),
        "high contrast": (
            "eq=contrast=1.15:"
            "saturation=1.10"
        ),
    }.get((eq_preset or "none").lower())

    if eq:

        parts.append(
            f"{current}{eq}[veq]"
        )

        current = "[veq]"

    if not parts:
        return "", current

    return (
        ";".join(parts),
        current,
    )


def _srt_timestamp_to_ass(ts: str) -> str:
    """'00:00:01,000' -> '0:00:01.00' (ASS format)."""

    m = re.match(
        r"(\d+):(\d+):(\d+)[,.](\d+)",
        (ts or "").strip(),
    )

    if not m:
        return "0:00:00.00"

    h, mi, s, ms = (int(x) for x in m.groups())

    return f"{h}:{mi:02d}:{s:02d}.{ms // 10:02d}"


def _parse_srt_cues(srt_text: str) -> list:
    """Parse SRT into (start, end, text) with ASS-safe text."""

    cues = []

    for block in re.split(
        r"\r?\n\s*\r?\n",
        (srt_text or "").strip(),
    ):

        lines = [
            ln
            for ln in block.splitlines()
            if ln.strip() != ""
        ]

        if not lines:
            continue

        if re.fullmatch(
            r"\d+", lines[0].strip()
        ):
            lines = lines[1:]

        if not lines or "-->" not in lines[0]:
            continue

        start_raw, end_raw = (
            p.strip()
            for p in lines[0].split("-->")
        )

        text = r"\N".join(lines[1:])
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("{", "\\{")
        text = text.replace("}", "\\}")

        cues.append(
            (
                _srt_timestamp_to_ass(start_raw),
                _srt_timestamp_to_ass(end_raw),
                text,
            )
        )

    return cues


def subtitles_filter_arg(
    srt_path: Path,
    font: str,
    size: int,
    position=88.0,
    color: str = "bright green",
    frame_w: int = 0,
    frame_h: int = 0,
):
    """Burn subtitles with a free vertical position.

    position is 0-100 (% of screen height, 0 = top,
    100 = bottom). Legacy "bottom"/"middle"/"top"
    strings still map to sensible percents.

    The SRT is converted to a styled ASS file with
    PlayRes set to the real frame size — the old
    force_style approach broke for large MarginV
    because libass defaults SRT to PlayResY=288 and
    pushed text off-screen.
    """

    import hashlib

    legacy = {
        "bottom": 88.0,
        "middle": 50.0,
        "top": 12.0,
    }

    if isinstance(position, str):
        pos = legacy.get(
            position.lower(), 88.0
        )
    else:
        try:
            pos = float(position)
        except (TypeError, ValueError):
            pos = 88.0

    pos = min(95.0, max(5.0, pos))

    fw = max(320, int(frame_w or 1280))
    fh = max(240, int(frame_h or 720))
    margin_v = int(
        round(fh * (100.0 - pos) / 100.0)
    )

    # ASS uses &HAABBGGRR
    color_key = (color or "bright green").lower()

    primary = {
        "bright green": "&H0000FF00",
        "white": "&H00FFFFFF",
        "yellow": "&H0000FFFF",
        "cyan": "&H00FFFF00",
    }.get(
        color_key,
        "&H0000FF00",
    )

    srt_path = Path(srt_path)
    srt_text = srt_path.read_text(
        encoding="utf-8", errors="replace"
    )
    cues = _parse_srt_cues(srt_text)

    # Style settings + content hash in the filename so
    # any change (text, size, color, position) yields a
    # fresh file — and busts the preview cache, since
    # the path is part of the filter string.

    digest = hashlib.md5(
        srt_text.encode("utf-8", "replace")
    ).hexdigest()[:8]

    safe_color = re.sub(
        r"[^a-z0-9]+", "", color_key
    )

    tag = (
        f"p{int(round(pos))}"
        f"s{int(size)}"
        f"{safe_color}"
        f"mv{margin_v}"
        f"{digest}"
    )

    ass_path = srt_path.with_name(
        f"{srt_path.stem}.{tag}.ass"
    )

    if not ass_path.exists():

        header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {fw}\n"
            f"PlayResY: {fh}\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, "
            "PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, "
            "MarginV, Encoding\n"
            f"Style: Sub,{font},{int(size)},{primary},"
            "&H000000FF,&H80000000,&H80000000,"
            "0,0,0,0,100,100,0,0,1,2,0,"
            f"2,10,10,{margin_v},1\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, "
            "MarginL, MarginR, MarginV, Effect, Text\n"
        )

        events = "".join(
            f"Dialogue: 0,{st},{en},Sub,,0,0,0,,"
            f"{tx}\n"
            for st, en, tx in cues
        )

        ass_path.write_text(
            header + events, encoding="utf-8"
        )

    return f"subtitles='{ass_path}'"


def ratio_filter_chain(
    start_label: str,
    ratio: str,
    bg_blur: int,
):
    """Convert aspect ratio; blurred-background variants included."""

    if ratio == "9:16 vertical (crop)":

        return (
            f"{start_label}"
            "scale=1080:1920:"
            "force_original_aspect_ratio=increase,"
            "crop=1080:1920"
            "[vratio]",
            "[vratio]",
        )

    if ratio in (
        "9:16 vertical (blur background)",
        "1:1 square (blur background)",
    ):

        out_w, out_h = (
            (1080, 1920)
            if ratio.startswith("9:16")
            else (1080, 1080)
        )

        radius = max(
            2,
            int(bg_blur),
        )

        parts = [
            f"{start_label}split=2[rbg][rfg]",
            f"[rbg]scale={out_w}:{out_h}:"
            "force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},"
            f"boxblur=luma_radius={radius}:"
            "luma_power=3[rbg2]",
            f"[rfg]scale={out_w}:-2[rfg2]",
            "[rbg2][rfg2]"
            "overlay=(W-w)/2:(H-h)/2"
            "[vratio]",
        ]

        return (
            ";".join(parts),
            "[vratio]",
        )

    return (
        f"{start_label}"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        "[vratio]",
        "[vratio]",
    )


def build_edit_video_filter(
    masks,
    mask_blur,
    burn_subs,
    srt_path,
    sub_font,
    sub_size,
    sub_pos,
    sub_color,
    ratio,
    bg_blur,
    flip=False,
    zoom_mode="off",
    zoom_amount=1.1,
    zoom_cx=50.0,
    zoom_cy=50.0,
    eq_preset="none",
    src_w=0,
    src_h=0,
    src_fps=30.0,
) -> str:
    """Full -vf chain: transforms -> masks -> burned subs -> ratio -> yuv420p."""

    filters = []
    current = "[0:v]"

    # Even frame dims: every downstream stage
    # (masks, subs) is computed against these.

    frame_w = (max(2, int(src_w)) // 2) * 2
    frame_h = (max(2, int(src_h)) // 2) * 2

    chain, current = transform_filter_chain(
        current,
        flip,
        zoom_mode,
        zoom_amount,
        eq_preset,
        src_w,
        src_h,
        src_fps,
        zoom_cx=zoom_cx,
        zoom_cy=zoom_cy,
    )

    if chain:
        filters.append(chain)

    if masks:

        chain, current = mask_filter_chain(
            current,
            masks,
            mask_blur,
            frame_w=frame_w,
            frame_h=frame_h,
        )

        filters.append(chain)

    if burn_subs and srt_path is not None:

        filters.append(
            f"{current}"
            + subtitles_filter_arg(
                srt_path,
                sub_font,
                sub_size,
                sub_pos,
                sub_color,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            + "[vsub]"
        )

        current = "[vsub]"

    chain, current = ratio_filter_chain(
        current,
        ratio,
        bg_blur,
    )

    filters.append(chain)

    filters.append(
        f"{current}format=yuv420p[vfinal]"
    )

    return ";".join(filters)


def make_title_card(
    text: str,
    duration: float,
    width: int,
    height: int,
    font: str,
    out_path: Path,
) -> Path:
    """Black title card (intro/outro) with centered text via libass.

    `text` comes straight from the user's UI input (their bytes),
    never retyped — Myanmar shaping stays intact.
    """

    dur = max(0.5, float(duration))
    w = (max(2, int(width)) // 2) * 2
    h = (max(2, int(height)) // 2) * 2

    safe = (text or "").replace("\\", "\\\\")

    ass = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1080\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, "
        "SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
        "Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Card,{font},64,&H0000FF00,&H000000FF,"
        "&H80000000,&H80000000,0,0,0,0,100,100,0,0,1,3,0,"
        "5,10,10,40,1\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, "
        "MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,9:59:59.99,Card,,0,0,0,,"
        f"{safe}\n"
    )

    ass_path = out_path.with_suffix(".ass")
    ass_path.write_text(ass, encoding="utf-8")

    vf = []

    if safe.strip():
        vf.append(f"subtitles='{ass_path}'")

    vf.append("format=yuv420p")

    result = run_cmd(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            (
                f"color=c=black:s={w}x{h}:"
                f"r=30:d={dur:.2f}"
            ),
            "-f",
            "lavfi",
            "-i",
            (
                "anullsrc=r=48000:cl=stereo"
                f":d={dur:.2f}"
            ),
            "-vf",
            ",".join(vf),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-shortest",
            str(out_path),
        ],
        timeout=600,
    )

    if (
        result.returncode != 0
        or not out_path.exists()
    ):

        raise RuntimeError(
            "Title card failed.\n"
            + (result.stderr or "")[-1000:]
        )

    return out_path


def split_video_parts(
    video_path: Path,
    part_len: float,
    base_name: str,
) -> list:
    """Split into ~part_len second chunks (stream copy)."""

    stem = Path(base_name).stem or "video"

    pattern = (
        EDIT_WORK_DIR
        / f"{stem}_part%02d.mp4"
    )

    # Clear stale parts from a previous run.
    for old in EDIT_WORK_DIR.glob(
        f"{stem}_part*.mp4"
    ):
        old.unlink(missing_ok=True)

    result = run_cmd(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-c",
            "copy",
            "-map",
            "0",
            "-f",
            "segment",
            "-segment_time",
            f"{float(part_len):.1f}",
            "-reset_timestamps",
            "1",
            str(pattern),
        ],
        timeout=1200,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Split failed.\n"
            + (result.stderr or "")[-1000:]
        )

    parts = sorted(
        EDIT_WORK_DIR.glob(
            f"{stem}_part*.mp4"
        )
    )

    return [
        p
        for p in parts
        if p.stat().st_size > 5000
    ]


def render_edited_video(
    video_path: Path,
    voice_path,
    keep_original: bool,
    orig_volume: float,
    masks: list,
    mask_blur: int,
    burn_subs: bool,
    srt_path,
    sub_font: str,
    sub_size: int,
    sub_pos: str,
    sub_color: str,
    ratio: str,
    bg_blur: int,
    out_name: str,
    flip: bool = False,
    zoom_mode: str = "off",
    zoom_amount: float = 1.1,
    zoom_cx: float = 50.0,
    zoom_cy: float = 50.0,
    eq_preset: str = "none",
    bgm_path=None,
    bgm_volume: float = 0.15,
    intro_text: str = "",
    outro_text: str = "",
    card_duration: float = 2.0,
) -> Path:

    info = probe_media(video_path)

    if not info["has_video"]:
        raise RuntimeError(
            "No video stream found in the file."
        )

    want_intro = bool((intro_text or "").strip())
    want_outro = bool((outro_text or "").strip())
    want_cards = want_intro or want_outro

    video_filter = build_edit_video_filter(
        masks,
        mask_blur,
        burn_subs,
        srt_path,
        sub_font,
        sub_size,
        sub_pos,
        sub_color,
        ratio,
        bg_blur,
        flip=flip,
        zoom_mode=zoom_mode,
        zoom_amount=zoom_amount,
        zoom_cx=zoom_cx,
        zoom_cy=zoom_cy,
        eq_preset=eq_preset,
        src_w=info["width"],
        src_h=info["height"],
        src_fps=info.get("fps", 30.0),
    )

    command = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
    ]

    voice_index = None

    if (
        voice_path is not None
        and Path(voice_path).exists()
    ):

        voice_index = 1

        command += [
            "-i",
            str(voice_path),
        ]

    bgm_index = None

    if (
        bgm_path is not None
        and Path(bgm_path).exists()
    ):

        bgm_index = (
            2
            if voice_index is not None
            else 1
        )

        command += [
            "-i",
            str(bgm_path),
        ]

    filter_parts = [video_filter]
    base_audio = None

    if voice_index is not None:

        if (
            keep_original
            and info["has_audio"]
        ):

            filter_parts.append(
                f"[0:a]volume={orig_volume:.3f}"
                "[a0];"
                f"[{voice_index}:a]aresample=48000"
                "[a1];"
                "[a0][a1]amix=inputs=2:"
                "duration=longest:"
                "dropout_transition=0"
                "[abase]"
            )

        else:

            filter_parts.append(
                f"[{voice_index}:a]"
                "aresample=48000"
                "[abase]"
            )

        base_audio = "[abase]"

    elif (
        keep_original
        and info["has_audio"]
    ):

        filter_parts.append(
            f"[0:a]volume={orig_volume:.3f}"
            "[abase]"
        )

        base_audio = "[abase]"

    if bgm_index is not None:

        bv = max(0.0, min(1.0, float(bgm_volume)))

        if base_audio is not None:

            filter_parts.append(
                f"[{bgm_index}:a]"
                f"volume={bv:.3f},"
                "aresample=48000,"
                "apad"
                "[abgm];"
                f"{base_audio}[abgm]"
                "amix=inputs=2:"
                "duration=first:"
                "dropout_transition=0"
                "[afinal]"
            )

        else:

            filter_parts.append(
                f"[{bgm_index}:a]"
                f"volume={bv:.3f},"
                "aresample=48000"
                "[afinal]"
            )

        audio_map = [
            "-map",
            "[afinal]",
        ]

    elif base_audio is not None:

        audio_map = [
            "-map",
            base_audio,
        ]

    else:

        audio_map = ["-an"]

    # Intro/outro concat needs an audio stream on every
    # segment, so force a silent track when the main
    # video would otherwise have none.

    if want_cards and audio_map == ["-an"]:

        silent_index = (
            1
            + (1 if voice_index is not None else 0)
            + (1 if bgm_index is not None else 0)
        )

        command += [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
        ]

        audio_map = [
            "-map",
            f"{silent_index}:a",
        ]

    main_name = (
        ("main_" + out_name)
        if want_cards
        else out_name
    )

    output = EDIT_WORK_DIR / main_name

    command += [
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[vfinal]",
        *audio_map,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output),
    ]

    result = run_cmd(
        command,
        timeout=3600,
    )

    if (
        result.returncode != 0
        or not output.exists()
        or output.stat().st_size < 5000
    ):

        raise RuntimeError(
            "Render failed.\n"
            + (result.stderr or "")[-2000:]
        )

    if not want_cards:
        return output

    main_info = probe_media(output)

    card_paths = []

    if want_intro:

        card_paths.append(
            make_title_card(
                intro_text,
                card_duration,
                main_info["width"] or 1080,
                main_info["height"] or 1920,
                sub_font,
                EDIT_WORK_DIR / "intro_card.mp4",
            )
        )

    card_paths.append(output)

    if want_outro:

        card_paths.append(
            make_title_card(
                outro_text,
                card_duration,
                main_info["width"] or 1080,
                main_info["height"] or 1920,
                sub_font,
                EDIT_WORK_DIR / "outro_card.mp4",
            )
        )

    concat_cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]

    for seg in card_paths:
        concat_cmd += ["-i", str(seg)]

    n = len(card_paths)

    # concat expects each segment's streams grouped:
    # [0:v][0:a][1:v][1:a]...
    seg_in = "".join(
        f"[{i}:v][{i}:a]"
        for i in range(n)
    )

    concat_cmd += [
        "-filter_complex",
        (
            f"{seg_in}"
            f"concat=n={n}:v=1:a=1"
            "[vcat][acat]"
        ),
        "-map",
        "[vcat]",
        "-map",
        "[acat]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(EDIT_WORK_DIR / out_name),
    ]

    concat_result = run_cmd(
        concat_cmd,
        timeout=3600,
    )

    final = EDIT_WORK_DIR / out_name

    if (
        concat_result.returncode != 0
        or not final.exists()
        or final.stat().st_size < 5000
    ):

        raise RuntimeError(
            "Intro/outro concat failed.\n"
            + (concat_result.stderr or "")[-2000:]
        )

    return final


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

if "step1_video_path" not in st.session_state:
    st.session_state.step1_video_path = ""


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

    speed_factor = st.slider(
        "⚡ Video speed (applied BEFORE transcription — "
        "SRT + voiceover stay in sync)",
        1.00,
        1.15,
        1.10,
        0.05,
        help=(
            "Slight speed-up helps avoid copyright "
            "detection. Applied to the video first, "
            "so transcription, SRT and voiceover "
            "are all timed to the sped-up video."
        ),
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

            save_uploaded_file(
                video_file,
                video_path,
            )

            # Optional speed-up BEFORE transcription so that
            # SRT timestamps + voiceover slots are all timed
            # to the sped-up video (nothing goes out of sync).

            if speed_factor > 1.001:

                sped_path = (
                    work
                    / "input_video_sped.mp4"
                )

                speed_result = run_cmd(
                    [
                        FFMPEG,
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(video_path),
                        "-vf",
                        (
                            "setpts="
                            f"PTS/{speed_factor:.3f}"
                        ),
                        "-af",
                        (
                            "atempo="
                            f"{speed_factor:.3f}"
                        ),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "20",
                        "-c:a",
                        "aac",
                        str(sped_path),
                    ],
                    timeout=1800,
                )

                if (
                    speed_result.returncode != 0
                    or not sped_path.exists()
                ):

                    raise RuntimeError(
                        "Speed-up failed.\n"
                        + (
                            speed_result.stderr
                            or ""
                        )[-1000:]
                    )

                video_path = sped_path

            # Keep a persistent copy for the Edit step
            # (the temp dir above is deleted afterwards).

            persist_video = (
                EDIT_WORK_DIR
                / (
                    "step1_video"
                    + (
                        Path(
                            video_file.name
                        ).suffix
                        or ".mp4"
                    )
                )
            )

            # Persist the PROCESSED video (after optional
            # speed-up), not the raw upload — the SRT timings
            # and voiceover slots are timed to this file, so
            # the Edit step must use the same one.

            shutil.copy(
                video_path,
                persist_video,
            )

            st.session_state.step1_video_path = str(
                persist_video
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

            (
                voice_path,
                segments,
                timing_report,
            ) = build_voiceover(
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
                gemini_client=(
                    get_gemini_client(
                        st.session_state.current_key_index
                    )
                ),
            )

            if timing_report["rewritten"]:

                # Keep the SRT in sync with the rewritten narration.
                st.session_state.srt_text = make_srt(
                    segments
                )

                st.info(
                    "AI rewrote "
                    f"{timing_report['rewritten']} "
                    "over-long lines shorter "
                    "(complete sentences, meaning kept). "
                    "The SRT above was updated to match."
                )

            if timing_report["overflow"]:

                total_over = round(
                    sum(
                        sec
                        for _, sec
                        in timing_report[
                            "overflow"
                        ]
                    ),
                    1,
                )

                st.warning(
                    f"{len(timing_report['overflow'])} "
                    "lines still run past their slots "
                    f"({total_over}s total). They were kept "
                    "at max 1.35x speed for intelligibility "
                    "and extend slightly into the following "
                    "pause instead of chipmunk audio."
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
# STEP 3 — EDIT & EXPORT
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    "③ Edit &amp; Export"
    "</div>",
    unsafe_allow_html=True,
)

c_prev, c_ctrl = st.columns([1, 1.15])

with c_ctrl:
    edit_video_file = st.file_uploader(
        "Video file",
        type=[
            "mp4",
            "mov",
            "mkv",
            "webm",
        ],
        key="edit_video",
    )

    use_step1_video = False

    if (
        st.session_state.step1_video_path
        and Path(
            st.session_state.step1_video_path
        ).exists()
    ):

        use_step1_video = st.checkbox(
            "Use the video from Step 1",
            value=True,
        )

    edit_video_path = None

    if use_step1_video:

        edit_video_path = Path(
            st.session_state.step1_video_path
        )

    elif edit_video_file:

        edit_video_path = (
            EDIT_WORK_DIR
            / (
                "edit_video"
                + (
                    Path(
                        edit_video_file.name
                    ).suffix
                    or ".mp4"
                )
            )
        )

        save_uploaded_file(
            edit_video_file,
            edit_video_path,
        )

    voiceover_choice = st.radio(
        "Voiceover track",
        [
            "Use Step 2 voiceover",
            "Upload audio file",
            "None",
        ],
        horizontal=True,
    )

    edit_voice_path = None

    if voiceover_choice == "Use Step 2 voiceover":

        if st.session_state.voice_bytes:

            edit_voice_path = (
                EDIT_WORK_DIR
                / "edit_voiceover.m4a"
            )

            edit_voice_path.write_bytes(
                st.session_state.voice_bytes
            )

        else:

            st.info(
                "No Step 2 voiceover yet — "
                "generate one above or upload a file."
            )

    elif voiceover_choice == "Upload audio file":

        voiceover_upload = st.file_uploader(
            "Voiceover audio",
            type=[
                "m4a",
                "mp3",
                "wav",
                "aac",
            ],
            key="edit_voiceover",
        )

        if voiceover_upload:

            edit_voice_path = (
                EDIT_WORK_DIR
                / (
                    "edit_voiceover_up"
                    + (
                        Path(
                            voiceover_upload.name
                        ).suffix
                        or ".m4a"
                    )
                )
            )

            edit_voice_path.write_bytes(
                voiceover_upload.getbuffer()
            )

    srt_choice = st.radio(
        "Subtitle source",
        [
            "Use Step 1 SRT",
            "Upload SRT file",
            "None",
        ],
        horizontal=True,
    )

    edit_srt_path = None
    edit_srt_available = False

    if srt_choice == "Use Step 1 SRT":

        if st.session_state.srt_text:

            edit_srt_path = (
                EDIT_WORK_DIR
                / "edit_subs.srt"
            )

            edit_srt_path.write_text(
                st.session_state.srt_text,
                encoding="utf-8-sig",
            )

            edit_srt_available = True

        else:

            st.info("No Step 1 SRT yet.")

    elif srt_choice == "Upload SRT file":

        srt_upload = st.file_uploader(
            "SRT file",
            type=["srt"],
            key="edit_srt",
        )

        if srt_upload:

            edit_srt_path = (
                EDIT_WORK_DIR
                / "edit_subs_up.srt"
            )

            edit_srt_path.write_bytes(
                srt_upload.getvalue()
            )

            edit_srt_available = True

    media_info = None
    edit_masks = []
    edit_mask_blur = 25
    sub_font = "sans-serif"
    burn_subs = False
    sub_size = 28
    sub_position = 88.0
    sub_color = "Bright green"
    out_ratio = "Original"
    bg_blur = 0
    flip_enabled = True
    zoom_choice = "Static zoom"
    zoom_amount = 1.10
    zoom_cx = 50.0
    zoom_cy = 50.0
    eq_choice = "Vivid"
    bgm_enabled = False
    edit_bgm_path = None
    bgm_volume = 0.15
    intro_text = ""
    outro_text = ""
    card_duration = 2.0
    split_enabled = True
    split_part_len = 120.0

    if (
        edit_video_path is not None
        and edit_video_path.exists()
    ):

        media_info = probe_media(
            edit_video_path
        )

        if not media_info["has_video"]:

            st.error(
                "No video stream found in the file."
            )

            st.stop()

        st.caption(
            f"{media_info['width']}x{media_info['height']} | "
            f"{media_info['duration']:.1f}s | "
            f"{'has audio' if media_info['has_audio'] else 'no audio'}"
        )

        st.subheader("Audio")

        original_choice = st.radio(
            "Original video audio",
            [
                "Mute original audio",
                "Keep original audio",
            ],
            horizontal=True,
        )

        original_volume = 0.0

        if (
            original_choice
            == "Keep original audio"
            and media_info["has_audio"]
        ):

            original_volume = (
                st.slider(
                    "Original audio volume",
                    0,
                    100,
                    40,
                )
                / 100.0
            )

        st.subheader(
            "Mask hardcoded subtitles"
        )

        mask_enabled = st.checkbox(
            "Cover burned-in subtitles with a mask",
            value=False,
        )

        edit_masks = []
        edit_mask_blur = 25

        if "edit_mask_list" not in st.session_state:
            st.session_state["edit_mask_list"] = []

        if "edit_mask_uid" not in st.session_state:
            st.session_state["edit_mask_uid"] = 0

        if mask_enabled:

            if st.button(
                "+ Add mask",
                key="add_mask_btn",
            ):

                st.session_state[
                    "edit_mask_uid"
                ] += 1

                st.session_state[
                    "edit_mask_list"
                ].append(
                    {
                        "uid": st.session_state[
                            "edit_mask_uid"
                        ],
                        "x": 0,
                        "y": 78,
                        "w": 100,
                        "h": 22,
                        "style": "Blur",
                    }
                )

                st.rerun()

            mask_list = st.session_state[
                "edit_mask_list"
            ]

            if not mask_list:

                st.info(
                    "No masks yet — tap "
                    "+ Add mask, then drag the "
                    "sliders to place each mask "
                    "freely."
                )

            for pos, m in enumerate(
                list(mask_list)
            ):

                uid = m["uid"]

                with st.expander(
                    f"Mask {pos + 1}",
                    expanded=(pos == 0),
                ):

                    c1, c2 = st.columns(2)

                    m["x"] = c1.slider(
                        "X (%)",
                        0,
                        100,
                        int(m["x"]),
                        key=f"mk_{uid}_x",
                    )
                    m["y"] = c2.slider(
                        "Y (%)",
                        0,
                        100,
                        int(m["y"]),
                        key=f"mk_{uid}_y",
                    )
                    m["w"] = c1.slider(
                        "Width (%)",
                        1,
                        100,
                        int(m["w"]),
                        key=f"mk_{uid}_w",
                    )
                    m["h"] = c2.slider(
                        "Height (%)",
                        1,
                        100,
                        int(m["h"]),
                        key=f"mk_{uid}_h",
                    )

                    m["style"] = st.radio(
                        "Style",
                        [
                            "Blur",
                            "Black box",
                        ],
                        index=(
                            0
                            if m["style"]
                            == "Blur"
                            else 1
                        ),
                        horizontal=True,
                        key=f"mk_{uid}_style",
                    )

                    if st.button(
                        "Remove this mask",
                        key=f"mk_{uid}_rm",
                    ):

                        st.session_state[
                            "edit_mask_list"
                        ] = [
                            mm
                            for mm in st.session_state[
                                "edit_mask_list"
                            ]
                            if mm["uid"] != uid
                        ]

                        st.rerun()

            if any(
                mm["style"] == "Blur"
                for mm in st.session_state[
                    "edit_mask_list"
                ]
            ):

                edit_mask_blur = (
                    st.slider(
                        "Mask blur strength",
                        1,
                        10,
                        5,
                    )
                    * 5
                )

            src_w = media_info["width"]
            src_h = media_info["height"]

            for mm in st.session_state[
                "edit_mask_list"
            ]:

                edit_masks.append(
                    {
                        "x": int(
                            src_w * mm["x"] / 100
                        ),
                        "y": int(
                            src_h * mm["y"] / 100
                        ),
                        "w": int(
                            src_w * mm["w"] / 100
                        ),
                        "h": int(
                            src_h * mm["h"] / 100
                        ),
                        "mode": (
                            "blur"
                            if mm["style"]
                            == "Blur"
                            else "black"
                        ),
                    }
                )

        st.subheader(
            "Copyright-safe transforms"
        )

        flip_enabled = st.checkbox(
            "Flip video horizontally",
            value=True,
        )

        zoom_choice = st.selectbox(
            "Zoom",
            [
                "Off",
                "Static zoom",
                "Slow push-in (dynamic)",
            ],
            index=1,
        )

        zoom_amount = 1.10
        zoom_cx = 50.0
        zoom_cy = 50.0

        if zoom_choice == "Static zoom":

            zoom_amount = st.slider(
                "Zoom amount",
                1.00,
                1.30,
                1.10,
                0.05,
            )

            zc1, zc2 = st.columns(2)

            zoom_cx = float(
                zc1.slider(
                    "Zoom center X (%)",
                    0,
                    100,
                    50,
                    help=(
                        "0 = left edge, "
                        "50 = center, "
                        "100 = right edge"
                    ),
                )
            )

            zoom_cy = float(
                zc2.slider(
                    "Zoom center Y (%)",
                    0,
                    100,
                    50,
                    help=(
                        "0 = top edge, "
                        "50 = center, "
                        "100 = bottom edge"
                    ),
                )
            )

        eq_choice = st.selectbox(
            "Color filter",
            [
                "None",
                "Vivid",
                "Warm",
                "Cool",
                "High contrast",
            ],
            index=1,
        )

        st.subheader("Subtitles")

        sub_font = find_myanmar_font()

        if not sub_font:

            st.warning(
                "No Myanmar font found on this server — "
                "burned subtitles may show as boxes. "
                "Install a Myanmar font (e.g. Noto Sans Myanmar) "
                "to fix it."
            )

            sub_font = "sans-serif"

        burn_subs = st.checkbox(
            "Burn subtitles into the video",
            value=edit_srt_available,
        )

        sub_size = st.slider(
            "Subtitle size",
            12,
            64,
            28,
        )

        sub_position = st.slider(
            "Subtitle vertical position (%)",
            5,
            95,
            88,
            help=(
                "0 = top of the screen, "
                "100 = bottom"
            ),
        )

        sub_color = st.selectbox(
            "Subtitle color",
            [
                "Bright green",
                "White",
                "Yellow",
                "Cyan",
            ],
            index=0,
        )

        st.subheader("Aspect ratio")

        out_ratio = st.selectbox(
            "Output ratio",
            [
                "Original",
                "9:16 vertical (blur background)",
                "9:16 vertical (crop)",
                "1:1 square (blur background)",
            ],
        )

        bg_blur = 0

        if "blur background" in out_ratio:

            bg_blur = (
                st.slider(
                    "Background blur strength",
                    1,
                    10,
                    6,
                )
                * 5
            )

        st.subheader("Background music")

        bgm_enabled = st.checkbox(
            "Add background music under the voiceover",
            value=False,
        )

        edit_bgm_path = None
        bgm_volume = 0.15

        if bgm_enabled:

            bgm_upload = st.file_uploader(
                "Music file",
                type=[
                    "mp3",
                    "wav",
                    "m4a",
                    "aac",
                ],
                key="edit_bgm",
            )

            if bgm_upload:

                edit_bgm_path = (
                    EDIT_WORK_DIR
                    / (
                        "edit_bgm"
                        + (
                            Path(
                                bgm_upload.name
                            ).suffix
                            or ".mp3"
                        )
                    )
                )

                edit_bgm_path.write_bytes(
                    bgm_upload.getbuffer()
                )

            bgm_volume = (
                st.slider(
                    "Music volume",
                    5,
                    30,
                    15,
                )
                / 100.0
            )

        st.subheader("Intro / outro cards")

        st.caption(
            "Leave empty to skip. Text is centered on "
            "a black card with a bright-green title."
        )

        intro_text = st.text_input(
            "Intro card text",
            value="",
        )

        outro_text = st.text_input(
            "Outro card text",
            value="",
        )

        card_duration = float(
            st.slider(
                "Card duration (seconds)",
                1,
                5,
                2,
            )
        )

        st.subheader("Auto-split")

        split_enabled = st.checkbox(
            "Split long videos into parts",
            value=True,
        )

        split_part_len = 120.0

        if split_enabled:

            split_part_len = float(
                st.slider(
                    "Part length (seconds)",
                    60,
                    300,
                    120,
                    10,
                )
            )


with c_prev:

    st.subheader("Live preview")

    if media_info:

        preview_time = st.slider(
            "Timestamp (seconds)",
            0.0,
            max(
                1.0,
                media_info["duration"],
            ),
            min(
                30.0,
                media_info["duration"] * 0.3,
            ),
            key="edit_preview_time",
        )

        try:

            preview_filter = (
                build_edit_video_filter(
                    edit_masks,
                    edit_mask_blur,
                    burn_subs
                    and edit_srt_available,
                    edit_srt_path,
                    sub_font,
                    sub_size,
                    sub_position,
                    sub_color,
                    out_ratio,
                    bg_blur,
                    flip=flip_enabled,
                    zoom_mode=(
                        "static"
                        if zoom_choice
                        == "Static zoom"
                        else (
                            "dynamic"
                            if zoom_choice
                            == "Slow push-in (dynamic)"
                            else "off"
                        )
                    ),
                    zoom_amount=zoom_amount,
                    zoom_cx=zoom_cx,
                    zoom_cy=zoom_cy,
                    eq_preset=eq_choice,
                    src_w=media_info["width"],
                    src_h=media_info["height"],
                    src_fps=media_info.get(
                        "fps", 30.0
                    ),
                )
            )

            try:

                video_stat = (
                    edit_video_path.stat()
                )

                cache_key = "|".join(
                    [
                        preview_filter,
                        f"{preview_time:.2f}",
                        str(video_stat.st_size),
                        str(
                            int(
                                video_stat.st_mtime
                            )
                        ),
                    ]
                )

            except OSError:

                cache_key = None

            preview_path = (
                EDIT_WORK_DIR
                / "preview.jpg"
            )

            if cache_key and (
                st.session_state.get(
                    "edit_preview_key"
                )
                != cache_key
                or not preview_path.exists()
            ):

                preview_result = run_cmd(
                    [
                        FFMPEG,
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{preview_time:.2f}",
                        "-i",
                        str(edit_video_path),
                        "-vframes",
                        "1",
                        "-vf",
                        preview_filter,
                        str(preview_path),
                    ],
                    timeout=120,
                )

                if (
                    preview_result.returncode
                    == 0
                    and preview_path.exists()
                ):

                    st.session_state[
                        "edit_preview_key"
                    ] = cache_key

                else:

                    preview_path.unlink(
                        missing_ok=True
                    )

                    st.error(
                        "Preview failed: "
                        + (
                            preview_result.stderr
                            or ""
                        )[:300]
                    )

                    st.session_state[
                        "edit_preview_key"
                    ] = cache_key

            if preview_path.exists():

                st.image(
                    str(preview_path),
                    caption=(
                        "Live preview at "
                        f"{preview_time:.1f}s"
                    ),
                )

        except Exception as exc:

            st.error(
                "Preview failed."
            )

            st.exception(exc)

    else:

        st.info(
            "Choose a video first — "
            "every change then updates "
            "this preview instantly, "
            "no button needed."
        )

    output_name = st.text_input(
        "Output filename",
        value="edited_video.mp4",
    )

    render_button = st.button(
        "Render final video",
        type="primary",
        use_container_width=True,
    )

    if render_button:

        file_name = safe_filename(
            output_name.strip(),
            "edited_video.mp4",
        )

        if not file_name.lower().endswith(
            ".mp4"
        ):

            file_name += ".mp4"

        try:

            with st.spinner(
                "Rendering video... "
                "this can take a few minutes."
            ):

                final_path = (
                    render_edited_video(
                        edit_video_path,
                        edit_voice_path,
                        keep_original=(
                            original_choice
                            == "Keep original audio"
                            and media_info[
                                "has_audio"
                            ]
                        ),
                        orig_volume=(
                            original_volume
                        ),
                        masks=edit_masks,
                        mask_blur=(
                            edit_mask_blur
                        ),
                        burn_subs=(
                            burn_subs
                            and edit_srt_available
                        ),
                        srt_path=(
                            edit_srt_path
                        ),
                        sub_font=sub_font,
                        sub_size=sub_size,
                        sub_pos=(
                            sub_position
                        ),
                        sub_color=sub_color,
                        ratio=out_ratio,
                        bg_blur=bg_blur,
                        out_name=file_name,
                        flip=flip_enabled,
                        zoom_mode=(
                            "static"
                            if zoom_choice
                            == "Static zoom"
                            else (
                                "dynamic"
                                if zoom_choice
                                == (
                                    "Slow push-in "
                                    "(dynamic)"
                                )
                                else "off"
                            )
                        ),
                        zoom_amount=zoom_amount,
                        zoom_cx=zoom_cx,
                        zoom_cy=zoom_cy,
                        eq_preset=eq_choice,
                        bgm_path=(
                            edit_bgm_path
                            if bgm_enabled
                            else None
                        ),
                        bgm_volume=bgm_volume,
                        intro_text=intro_text,
                        outro_text=outro_text,
                        card_duration=(
                            card_duration
                        ),
                    )
                )

            final_info = probe_media(
                final_path
            )

            do_split = (
                split_enabled
                and final_info["duration"]
                > split_part_len
            )

            if do_split:

                with st.spinner(
                    "Splitting into parts..."
                ):

                    parts = split_video_parts(
                        final_path,
                        split_part_len,
                        file_name,
                    )

                if not parts:

                    st.warning(
                        "Split produced no parts — "
                        "showing the full video."
                    )

                    do_split = False

            if do_split:

                st.success(
                    f"Done — {len(parts)} parts."
                )

                for idx, part in enumerate(
                    parts
                ):
                    st.markdown(
                        f"**အပိုင်း {idx + 1}**"
                    )

                    st.video(
                        str(part)
                    )

                    st.download_button(
                        f"Download part {idx + 1}",
                        data=(
                            part.read_bytes()
                        ),
                        file_name=(
                            part.name
                        ),
                        mime="video/mp4",
                        use_container_width=True,
                        key=(
                            f"dl_part_{idx}"
                        ),
                    )

            else:

                st.video(
                    str(final_path)
                )

                st.download_button(
                    "Download final video",
                    data=(
                        final_path.read_bytes()
                    ),
                    file_name=(
                        final_path.name
                    ),
                    mime="video/mp4",
                    use_container_width=True,
                )

        except Exception as exc:

            st.error(
                "Render failed."
            )

            st.exception(exc)


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

