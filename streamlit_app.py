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
    layout="centered",
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================
# MODELS / VOICES
# ============================================================

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

# Microsoft Burmese currently has these two standard voices.
BASE_VOICES = {
    "သီဟ (အမျိုးသား)": "my-MM-ThihaNeural",
    "နီလာ (အမျိုးသမီး)": "my-MM-NilarNeural",
}

# Character profiles.
# These do NOT pretend to be new Burmese voices.
# They modify rate/pitch/volume around the real Burmese voices.
VOICE_PROFILES = {
    "သီဟ • ပုံမှန်": {
        "voice": "my-MM-ThihaNeural",
        "rate": 0,
        "pitch": 0,
        "volume": 1.00,
    },
    "သီဟ • လူငယ်": {
        "voice": "my-MM-ThihaNeural",
        "rate": 7,
        "pitch": 4,
        "volume": 1.00,
    },
    "သီဟ • နက်နက်": {
        "voice": "my-MM-ThihaNeural",
        "rate": -5,
        "pitch": -10,
        "volume": 1.00,
    },
    "နီလာ • ပုံမှန်": {
        "voice": "my-MM-NilarNeural",
        "rate": 0,
        "pitch": 0,
        "volume": 1.00,
    },
    "နီလာ • ပျော့ပျောင်း": {
        "voice": "my-MM-NilarNeural",
        "rate": -4,
        "pitch": 5,
        "volume": 1.00,
    },
    "နီလာ • တက်ကြွ": {
        "voice": "my-MM-NilarNeural",
        "rate": 8,
        "pitch": 3,
        "volume": 1.00,
    },
}

TRANSLATION_STYLES = {
    "🎬 Movie Natural": (
        "Use natural spoken Burmese suitable for professional "
        "movie dubbing. Preserve emotion and character personality."
    ),
    "😂 Casual / Funny": (
        "Use conversational Burmese. Preserve humor, sarcasm, "
        "teasing and casual character personality when present."
    ),
    "📖 Standard": (
        "Use clear standard Burmese while preserving the exact "
        "meaning and emotional tone."
    ),
}

MAX_CHARS = 42
MAX_DURATION = 6.0
PAUSE_SPLIT = 0.65
MIN_SEGMENT_DURATION = 0.35


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "generated_srt": None,
    "generated_voice": None,
    "voice_preview": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at top left,
            rgba(90, 50, 140, 0.30),
            transparent 35%
        ),
        radial-gradient(
            circle at top right,
            rgba(30, 100, 160, 0.22),
            transparent 35%
        ),
        #080a10;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 28px;
    border-radius: 22px;
    margin-bottom: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(30, 32, 48, 0.95),
            rgba(14, 17, 26, 0.95)
        );
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 40px rgba(0,0,0,0.25);
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 6px;
}

.hero-sub {
    color: #aeb4c0;
    font-size: 15px;
}

.card {
    padding: 24px;
    border-radius: 20px;
    background: rgba(20,23,32,0.88);
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 20px;
}

.card-title {
    font-size: 22px;
    font-weight: 750;
    margin-bottom: 5px;
}

.card-sub {
    color: #9da5b2;
    font-size: 14px;
    margin-bottom: 18px;
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
        if value:
            return str(value).strip()
    except Exception:
        pass

    return os.environ.get(name, "").strip()


def get_gemini_client():
    key = get_secret("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ကို စစ်ပါ။"
        )

    return genai.Client(api_key=key)


def get_deepgram_key():
    key = get_secret("DEEPGRAM_API_KEY")

    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ကို စစ်ပါ။"
        )

    return key


def safe_filename(name, default="output"):
    name = (name or "").strip()

    if not name:
        name = default

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

    return name[:120]


def run_cmd(cmd, error_text="FFmpeg error"):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{error_text}\n\n"
            f"{result.stderr[-5000:]}"
        )

    return result


# ============================================================
# DURATION
# ============================================================

def probe_duration(path):
    """
    Uses FFmpeg itself instead of assuming ffprobe exists.
    """

    result = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    text = result.stderr

    match = re.search(
        r"Duration:\s*(\d{2}):(\d{2}):(\d{2})\.(\d+)",
        text,
    )

    if not match:
        return 0.0

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))

    fraction = match.group(4)

    try:
        fraction_value = float(
            "0." + fraction
        )
    except Exception:
        fraction_value = 0.0

    return (
        hours * 3600
        + minutes * 60
        + seconds
        + fraction_value
    )


# ============================================================
# TIME
# ============================================================

def parse_seconds(value):
    if value is None:
        return 0.0

    text = str(value).strip()

    if text.endswith("s"):
        text = text[:-1]

    try:
        return float(text)
    except Exception:
        return 0.0


def srt_time(seconds):
    seconds = max(
        0.0,
        float(seconds),
    )

    total_ms = int(
        round(seconds * 1000)
    )

    hours = total_ms // 3_600_000
    total_ms %= 3_600_000

    minutes = total_ms // 60_000
    total_ms %= 60_000

    secs = total_ms // 1000
    milliseconds = total_ms % 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


# ============================================================
# VIDEO -> AUDIO
# ============================================================

def extract_audio(
    video_path,
    output_path,
):
    run_cmd(
        [
            FFMPEG,
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
            str(output_path),
        ],
        "❌ Video ကနေ audio ခွဲမရပါ",
    )


# ============================================================
# DEEPGRAM
# ============================================================

def transcribe_deepgram(audio_path):
    api_key = get_deepgram_key()

    url = (
        "https://api.deepgram.com/v1/listen"
        "?model=nova-3"
        "&detect_language=true"
        "&punctuate=true"
        "&smart_format=true"
        "&utterances=true"
        "&diarize=true"
        "&words=true"
    )

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }

    with open(
        audio_path,
        "rb",
    ) as audio:
        response = requests.post(
            url,
            headers=headers,
            data=audio,
            timeout=1800,
        )

    if response.status_code != 200:
        raise RuntimeError(
            "Deepgram transcription failed:\n"
            + response.text[:3000]
        )

    return response.json()


# ============================================================
# WORD EXTRACTION
# ============================================================

def extract_words(data):
    words = []

    channels = (
        data
        .get("results", {})
        .get("channels", [])
    )

    for channel in channels:
        alternatives = channel.get(
            "alternatives",
            [],
        )

        if not alternatives:
            continue

        alternative = alternatives[0]

        for word in alternative.get(
            "words",
            [],
        ):
            text = str(
                word.get("punctuated_word")
                or word.get("word")
                or ""
            ).strip()

            if not text:
                continue

            start = parse_seconds(
                word.get("start")
            )

            end = parse_seconds(
                word.get("end")
            )

            if end <= start:
                continue

            words.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "speaker": word.get(
                        "speaker"
                    ),
                }
            )

    return sorted(
        words,
        key=lambda x: x["start"],
    )


# ============================================================
# UTTERANCE FALLBACK
# ============================================================

def extract_utterances(data):
    result = (
        data
        .get("results", {})
    )

    utterances = result.get(
        "utterances",
        [],
    )

    output = []

    for item in utterances:
        text = str(
            item.get("transcript")
            or ""
        ).strip()

        start = parse_seconds(
            item.get("start")
        )

        end = parse_seconds(
            item.get("end")
        )

        if (
            text
            and end > start
        ):
            output.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "speaker": item.get(
                        "speaker"
                    ),
                }
            )

    return output


# ============================================================
# SMART SEGMENTATION
# ============================================================

def build_segments(words):
    if not words:
        return []

    segments = []
    current = []

    sentence_endings = (
        ".",
        "?",
        "!",
        "。",
        "？",
        "！",
        "…",
    )

    def flush():
        nonlocal current

        if not current:
            return

        text = " ".join(
            x["text"]
            for x in current
        ).strip()

        start = current[0]["start"]
        end = current[-1]["end"]

        if (
            text
            and end - start >= MIN_SEGMENT_DURATION
        ):
            segments.append(
                {
                    "id": len(segments) + 1,
                    "start": round(
                        start,
                        3,
                    ),
                    "end": round(
                        end,
                        3,
                    ),
                    "duration": round(
                        end - start,
                        3,
                    ),
                    "text": text,
                    "speaker": current[0].get(
                        "speaker"
                    ),
                    "myanmar": "",
                }
            )

        current = []

    for word in words:

        if not current:
            current = [word]
            continue

        previous = current[-1]

        gap = (
            word["start"]
            - previous["end"]
        )

        current_text = " ".join(
            x["text"]
            for x in current
        )

        proposed_text = (
            current_text
            + " "
            + word["text"]
        ).strip()

        duration = (
            word["end"]
            - current[0]["start"]
        )

        speaker_changed = (
            current[0].get("speaker")
            is not None
            and word.get("speaker")
            is not None
            and current[0]["speaker"]
            != word["speaker"]
        )

        sentence_finished = (
            current_text.endswith(
                sentence_endings
            )
        )

        should_split = False

        if gap >= PAUSE_SPLIT:
            should_split = True

        if speaker_changed:
            should_split = True

        if (
            duration > MAX_DURATION
            and len(current) >= 2
        ):
            should_split = True

        if (
            len(proposed_text)
            > MAX_CHARS
            and len(current) >= 2
        ):
            should_split = True

        if sentence_finished:
            should_split = True

        if should_split:
            flush()
            current = [word]
        else:
            current.append(word)

    flush()

    return segments


def segments_from_utterances(utterances):
    segments = []

    for item in utterances:
        text = item["text"].strip()

        if not text:
            continue

        start = item["start"]
        end = item["end"]

        # Split very long utterances.
        words = text.split()

        if (
            len(text) <= MAX_CHARS
            and end - start <= MAX_DURATION
        ):
            segments.append(
                {
                    "id": len(segments) + 1,
                    "start": round(
                        start,
                        3,
                    ),
                    "end": round(
                        end,
                        3,
                    ),
                    "duration": round(
                        end - start,
                        3,
                    ),
                    "text": text,
                    "speaker": item.get(
                        "speaker"
                    ),
                    "myanmar": "",
                }
            )
            continue

        total_duration = max(
            0.1,
            end - start,
        )

        chunk = []
        chunk_start = start

        for word in words:
            chunk.append(word)

            proposed = " ".join(chunk)

            estimated = (
                total_duration
                * len(proposed)
                / max(len(text), 1)
            )

            if (
                len(proposed) >= MAX_CHARS
                or estimated >= MAX_DURATION
            ):
                ratio = (
                    len(proposed)
                    / max(len(text), 1)
                )

                chunk_end = min(
                    end,
                    chunk_start
                    + total_duration
                    * ratio,
                )

                segments.append(
                    {
                        "id": len(segments) + 1,
                        "start": round(
                            chunk_start,
                            3,
                        ),
                        "end": round(
                            chunk_end,
                            3,
                        ),
                        "duration": round(
                            chunk_end
                            - chunk_start,
                            3,
                        ),
                        "text": proposed,
                        "speaker": item.get(
                            "speaker"
                        ),
                        "myanmar": "",
                    }
                )

                chunk = []
                chunk_start = chunk_end

        if chunk:
            segments.append(
                {
                    "id": len(segments) + 1,
                    "start": round(
                        chunk_start,
                        3,
                    ),
                    "end": round(
                        end,
                        3,
                    ),
                    "duration": round(
                        end
                        - chunk_start,
                        3,
                    ),
                    "text": " ".join(chunk),
                    "speaker": item.get(
                        "speaker"
                    ),
                    "myanmar": "",
                }
            )

    return segments


# ============================================================
# GEMINI
# ============================================================

def clean_json(text):
    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip()
            == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    return text


def translation_prompt(
    items,
    mode,
):
    style_instruction = (
        TRANSLATION_STYLES.get(
            mode,
            TRANSLATION_STYLES[
                "🎬 Movie Natural"
            ],
        )
    )

    return f"""
You are a professional Burmese movie dubbing translator.

TASK:
Translate each English/foreign-language dialogue into natural
spoken Myanmar Burmese.

STYLE:
{style_instruction}

CRITICAL RULES:
1. Translate EVERY ID.
2. Do not skip any dialogue.
3. Do not summarize.
4. Do not explain.
5. Do not add information.
6. Preserve emotion, intention and character personality.
7. Burmese should sound natural when spoken aloud.
8. Avoid stiff word-for-word translation.
9. Keep names consistent.
10. Keep the translation reasonably short for dubbing.
11. Do not translate the ID.
12. Return ONLY valid JSON.

CONTEXT:
The items are consecutive movie dialogue. Use nearby lines
to understand pronouns, relationships and meaning.

INPUT:
{json.dumps(
    items,
    ensure_ascii=False,
    indent=2,
)}

OUTPUT:
[
  {{
    "id": 1,
    "myanmar": "မြန်မာဘာသာပြန်"
  }}
]
"""


def translate_batch(
    client,
    batch,
    mode,
):
    prompt = translation_prompt(
        batch,
        mode,
    )

    last_error = None

    for model in GEMINI_MODELS:

        for attempt in range(2):

            try:
                response = (
                    client
                    .models
                    .generate_content(
                        model=model,
                        contents=prompt,
                    )
                )

                raw = clean_json(
                    getattr(
                        response,
                        "text",
                        "",
                    )
                )

                if not raw:
                    raise ValueError(
                        "Gemini response empty"
                    )

                data = json.loads(
                    raw
                )

                if not isinstance(
                    data,
                    list,
                ):
                    raise ValueError(
                        "Gemini JSON မမှန်ပါ"
                    )

                result = {}

                for item in data:

                    if (
                        "id" not in item
                        or "myanmar"
                        not in item
                    ):
                        continue

                    result[
                        int(item["id"])
                    ] = str(
                        item["myanmar"]
                    ).strip()

                expected = {
                    int(x["id"])
                    for x in batch
                }

                if (
                    set(result.keys())
                    != expected
                ):
                    raise ValueError(
                        "Translation result "
                        "မပြည့်စုံပါ"
                    )

                return result

            except Exception as exc:

                last_error = exc

                if attempt == 0:
                    time.sleep(
                        1.5
                        + random.random()
                    )
                else:
                    time.sleep(
                        0.8
                    )

    raise RuntimeError(
        "Gemini Translation မအောင်မြင်ပါ။\n\n"
        f"နောက်ဆုံး error: {last_error}"
    )


def translate_segments(
    segments,
    mode,
    progress,
):
    client = get_gemini_client()

    batch_size = 8
    total = len(segments)

    for start in range(
        0,
        total,
        batch_size,
    ):
        batch = segments[
            start:
            start + batch_size
        ]

        items = []

        # Give Gemini a small context window.
        context_start = max(
            0,
            start - 2,
        )

        context_end = min(
            total,
            start
            + batch_size
            + 2,
        )

        context_items = segments[
            context_start:
            context_end
        ]

        for segment in context_items:
            items.append(
                {
                    "id": segment["id"],
                    "text": segment["text"],
                    "translate": (
                        start
                        <= segment["id"] - 1
                        < start + batch_size
                    ),
                }
            )

        # Only requested batch is returned.
        # Context lines are marked translate=false.
        prompt = translation_prompt(
            items,
            mode,
        )

        # Replace function's prompt with explicit context-aware prompt.
        prompt = f"""
You are a professional movie dubbing translator.

Translate ONLY the dialogue items where "translate" is true.

STYLE:
{TRANSLATION_STYLES.get(
    mode,
    TRANSLATION_STYLES["🎬 Movie Natural"],
)}

RULES:
- Use nearby context to understand pronouns and relationships.
- Preserve emotion and character personality.
- Use natural spoken Myanmar Burmese.
- Do not translate context-only items.
- Do not summarize.
- Do not explain.
- Do not add information.
- Keep names consistent.
- Keep lines suitable for voice dubbing.
- Return ONLY JSON for the requested items.
- Every requested ID must appear exactly once.

DIALOGUE:
{json.dumps(
    items,
    ensure_ascii=False,
    indent=2,
)}

OUTPUT:
[
  {{
    "id": 1,
    "myanmar": "..."
  }}
]
"""

        last_error = None
        result = None

        for model in GEMINI_MODELS:

            for attempt in range(2):

                try:
                    response = (
                        client
                        .models
                        .generate_content(
                            model=model,
                            contents=prompt,
                        )
                    )

                    raw = clean_json(
                        getattr(
                            response,
                            "text",
                            "",
                        )
                    )

                    data = json.loads(
                        raw
                    )

                    if not isinstance(
                        data,
                        list,
                    ):
                        raise ValueError(
                            "Invalid JSON"
                        )

                    result = {}

                    for item in data:

                        if (
                            "id" not in item
                            or "myanmar"
                            not in item
                        ):
                            continue

                        result[
                            int(item["id"])
                        ] = str(
                            item["myanmar"]
                        ).strip()

                    expected = {
                        x["id"]
                        for x in batch
                    }

                    if (
                        set(result)
                        != expected
                    ):
                        raise ValueError(
                            "Translation incomplete"
                        )

                    break

                except Exception as exc:
                    last_error = exc

                    if attempt == 0:
                        time.sleep(
                            1.5
                            + random.random()
                        )
                    else:
                        time.sleep(
                            0.8
                        )

            if result is not None:
                break

        if result is None:
            raise RuntimeError(
                "Gemini Translation မအောင်မြင်ပါ။\n\n"
                f"{last_error}"
            )

        for segment in batch:
            segment["myanmar"] = result[
                segment["id"]
            ]

        progress(
            min(
                1.0,
                (start + len(batch))
                / max(total, 1),
            ),
            f"🇲🇲 ဘာသာပြန်ပြီးပါပြီ "
            f"{min(start + len(batch), total)}/{total}",
        )

    return segments


# ============================================================
# SRT
# ============================================================

def make_srt(segments):
    lines = []

    for index, segment in enumerate(
        segments,
        1,
    ):
        text = (
            segment.get("myanmar")
            or segment.get("text")
            or ""
        ).strip()

        lines.append(
            str(index)
        )

        lines.append(
            f"{srt_time(segment['start'])} --> "
            f"{srt_time(segment['end'])}"
        )

        lines.append(text)
        lines.append("")

    return "\n".join(lines)


SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    r"\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def parse_srt_time(text):
    match = re.match(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})",
        text.strip(),
    )

    if not match:
        return None

    h, m, s, ms = map(
        int,
        match.groups(),
    )

    return (
        h * 3600
        + m * 60
        + s
        + ms / 1000
    )


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

    entries = []

    for block in blocks:

        lines = block.splitlines()

        if len(lines) < 2:
            continue

        timing_index = None

        for i, line in enumerate(lines):

            if "-->" in line:
                timing_index = i
                break

        if timing_index is None:
            continue

        timing = lines[
            timing_index
        ].strip()

        match = SRT_TIME_RE.match(
            timing
        )

        if not match:
            continue

        parts = match.group(
            0
        ).split("-->")

        start = parse_srt_time(
            parts[0]
        )

        end = parse_srt_time(
            parts[1]
        )

        if (
            start is None
            or end is None
            or end <= start
        ):
            continue

        subtitle = "\n".join(
            lines[
                timing_index + 1:
            ]
        ).strip()

        if not subtitle:
            continue

        entries.append(
            {
                "start": start,
                "end": end,
                "text": subtitle,
            }
        )

    entries.sort(
        key=lambda x: x["start"]
    )

    return entries


# ============================================================
# SRT QUALITY / AUTO FIX
# ============================================================

def quality_check(entries):
    issues = []

    previous_end = 0.0

    for index, item in enumerate(
        entries,
        1,
    ):
        start = item["start"]
        end = item["end"]
        text = item["text"]

        duration = end - start

        if end <= start:
            issues.append(
                f"#{index}: duration မမှန်"
            )

        if start < previous_end:
            issues.append(
                f"#{index}: timestamp overlap"
            )

        if duration > 8.0:
            issues.append(
                f"#{index}: subtitle duration ရှည်လွန်း"
            )

        clean_text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if len(clean_text) > 84:
            issues.append(
                f"#{index}: subtitle စာရှည်လွန်း"
            )

        previous_end = max(
            previous_end,
            end,
        )

    return issues


def auto_fix_srt(entries):
    fixed = 0

    for i in range(
        len(entries)
    ):
        current = entries[i]

        # Invalid duration.
        if current["end"] <= current["start"]:
            current["end"] = (
                current["start"]
                + 0.5
            )
            fixed += 1

        # Prevent overlap with next subtitle.
        if (
            i + 1
            < len(entries)
        ):
            next_item = entries[
                i + 1
            ]

            if (
                current["end"]
                > next_item["start"]
            ):
                new_end = max(
                    current["start"]
                    + 0.05,
                    next_item["start"],
                )

                if (
                    new_end
                    != current["end"]
                ):
                    current["end"] = (
                        new_end
                    )
                    fixed += 1

    return fixed


# ============================================================
# TTS
# ============================================================

async def _tts(
    text,
    path,
    voice,
    rate,
    pitch,
):
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=f"{rate:+d}%",
        pitch=f"{pitch:+d}Hz",
    )

    await communicator.save(
        path
    )


def create_tts(
    text,
    path,
    voice,
    rate,
    pitch,
):
    try:
        asyncio.run(
            _tts(
                text,
                path,
                voice,
                rate,
                pitch,
            )
        )
    except RuntimeError as exc:

        if "asyncio.run()" in str(exc):
            loop = asyncio.new_event_loop()

            try:
                loop.run_until_complete(
                    _tts(
                        text,
                        path,
                        voice,
                        rate,
                        pitch,
                    )
                )
            finally:
                loop.close()
        else:
            raise


# ============================================================
# ATEMPO CHAIN
# ============================================================

def make_atempo_filter(speed):
    """
    FFmpeg atempo accepts 0.5 - 2.0 per filter.
    Chain multiple filters for larger corrections.
    """

    speed = max(
        0.25,
        min(
            4.0,
            float(speed),
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


# ============================================================
# AUDIO FIT / VOLUME
# ============================================================

def fit_audio_to_slot(
    voice_path,
    slot_duration,
    output_path,
    user_speed=1.0,
):
    raw_duration = probe_duration(
        voice_path
    )

    if raw_duration <= 0:
        raise RuntimeError(
            "Voice duration ဖတ်မရပါ"
        )

    slot_duration = max(
        0.1,
        float(slot_duration),
    )

    # raw / slot:
    # >1 means voice is too long -> speed up
    # <1 means voice is short -> slow down
    fit_speed = (
        raw_duration
        / slot_duration
    )

    final_speed = (
        fit_speed
        * float(user_speed)
    )

    # Keep extreme changes reasonable.
    final_speed = max(
        0.25,
        min(
            4.0,
            final_speed,
        ),
    )

    tempo_filter = (
        make_atempo_filter(
            final_speed
        )
    )

    filters = [
        tempo_filter,
        "asetpts=PTS-STARTPTS",
        f"atrim=0:{slot_duration:.6f}",
        "apad",
        f"atrim=0:{slot_duration:.6f}",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "alimiter=limit=0.95",
    ]

    run_cmd(
        [
            FFMPEG,
            "-y",
            "-i",
            str(voice_path),
            "-filter:a",
            ",".join(filters),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(output_path),
        ],
        "Voice timing ပြင်မရပါ",
    )


# ============================================================
# BUILD VOICEOVER
# ============================================================

def build_voiceover(
    entries,
    profile_name,
    user_speed,
    progress,
):
    if not entries:
        raise RuntimeError(
            "SRT ထဲမှာ subtitle မတွေ့ပါ"
        )

    profile = VOICE_PROFILES[
        profile_name
    ]

    workdir = tempfile.mkdtemp(
        prefix="myanmar_voice_"
    )

    raw_dir = os.path.join(
        workdir,
        "raw",
    )

    fitted_dir = os.path.join(
        workdir,
        "fitted",
    )

    os.makedirs(
        raw_dir,
        exist_ok=True,
    )

    os.makedirs(
        fitted_dir,
        exist_ok=True,
    )

    fitted_files = []

    total = len(entries)

    for i, entry in enumerate(
        entries
    ):
        text = (
            entry["text"]
            .replace("\n", " ")
            .strip()
        )

        if not text:
            continue

        raw_path = os.path.join(
            raw_dir,
            f"voice_{i:05d}.mp3",
        )

        fitted_path = os.path.join(
            fitted_dir,
            f"voice_{i:05d}.m4a",
        )

        create_tts(
            text,
            raw_path,
            profile["voice"],
            profile["rate"],
            profile["pitch"],
        )

        slot = max(
            0.1,
            entry["end"]
            - entry["start"],
        )

        fit_audio_to_slot(
            raw_path,
            slot,
            fitted_path,
            user_speed,
        )

        fitted_files.append(
            (
                fitted_path,
                entry["start"],
            )
        )

        progress(
            (i + 1)
            / max(total, 1),
            f"🎙️ Voice {i + 1}/{total}",
        )

    if not fitted_files:
        raise RuntimeError(
            "Voice audio မထုတ်နိုင်ပါ"
        )

    output = os.path.join(
        workdir,
        "Myanmar_Voiceover.m4a",
    )

    inputs = []
    filters = []
    labels = []

    for i, (
        audio_path,
        start,
    ) in enumerate(
        fitted_files
    ):
        inputs.extend(
            [
                "-i",
                audio_path,
            ]
        )

        delay = max(
            0,
            int(
                round(
                    start * 1000
                )
            ),
        )

        label = f"a{i}"

        filters.append(
            f"[{i}:a]"
            f"adelay={delay}|{delay}"
            f"[{label}]"
        )

        labels.append(
            f"[{label}]"
        )

    # Final mix.
    filters.append(
        "".join(labels)
        + f"amix="
          f"inputs={len(labels)}:"
          "duration=longest:"
          "dropout_transition=0:"
          "normalize=0,"
          "volume=1.35,"
          "loudnorm="
          "I=-16:"
          "TP=-1.5:"
          "LRA=11,"
          "alimiter=limit=0.95,"
          "aresample=48000"
          "[mix]"
    )

    command = [
        FFMPEG,
        "-y",
    ]

    command.extend(
        inputs
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[mix]",
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
            output,
        ]
    )

    run_cmd(
        command,
        "Myanmar Voiceover မထုတ်နိုင်ပါ",
    )

    return output


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-title">🎬 Myanmar Movie AI</div>
    <div class="hero-sub">
        AI Subtitle Translation & Myanmar Voiceover Studio
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# STEP 1
# ============================================================

st.markdown(
    """
<div class="card">
    <div class="card-title">
        ① Video → မြန်မာ SRT
    </div>
    <div class="card-sub">
        Dialogue ကို auto detect လုပ်ပြီး
        context-aware မြန်မာဘာသာပြန်ပေးမယ်။
    </div>
</div>
""",
    unsafe_allow_html=True,
)

video_file = st.file_uploader(
    "🎥 Video Upload",
    type=[
        "mp4",
        "mkv",
        "mov",
        "avi",
        "webm",
        "m4v",
    ],
    key="main_video",
)

with st.form("srt_form"):

    translation_mode = st.selectbox(
        "🎬 Translation Style",
        list(
            TRANSLATION_STYLES.keys()
        ),
    )

    generate_srt = st.form_submit_button(
        "📝 မြန်မာ SRT ထုတ်မယ်",
        type="primary",
        use_container_width=True,
    )


if generate_srt:

    if video_file is None:

        st.error(
            "🎥 Video အရင် Upload လုပ်ပါ။"
        )

    else:

        try:

            with st.status(
                "🚀 SRT processing စနေပါတယ်...",
                expanded=True,
            ) as status:

                workdir = tempfile.mkdtemp(
                    prefix="movie_srt_"
                )

                video_path = os.path.join(
                    workdir,
                    safe_filename(
                        video_file.name,
                        "video.mp4",
                    ),
                )

                with open(
                    video_path,
                    "wb",
                ) as f:
                    f.write(
                        video_file.getbuffer()
                    )

                audio_path = os.path.join(
                    workdir,
                    "audio.wav",
                )

                st.write(
                    "🎧 Audio ခွဲနေပါတယ်..."
                )

                extract_audio(
                    video_path,
                    audio_path,
                )

                st.write(
                    "🧠 Deepgram dialogue "
                    "transcription..."
                )

                data = transcribe_deepgram(
                    audio_path
                )

                words = extract_words(
                    data
                )

                if words:

                    segments = build_segments(
                        words
                    )

                else:

                    st.write(
                        "ℹ️ Word timestamps "
                        "မရပါ။ Utterance fallback သုံးနေပါတယ်..."
                    )

                    utterances = (
                        extract_utterances(
                            data
                        )
                    )

                    segments = (
                        segments_from_utterances(
                            utterances
                        )
                    )

                if not segments:
                    raise RuntimeError(
                        "Dialogue segments မတွေ့ပါ။ "
                        "Audio ကို စစ်ကြည့်ပါ။"
                    )

                st.write(
                    f"📝 Dialogue "
                    f"{len(segments)} ခု တွေ့ပါပြီ"
                )

                progress_bar = st.progress(
                    0
                )

                def translation_progress(
                    value,
                    text,
                ):
                    progress_bar.progress(
                        min(
                            1.0,
                            max(
                                0.0,
                                float(value),
                            ),
                        )
                    )

                    st.write(text)

                segments = (
                    translate_segments(
                        segments,
                        translation_mode,
                        translation_progress,
                    )
                )

                srt_text = make_srt(
                    segments
                )

                st.session_state.generated_srt = (
                    srt_text
                )

                status.update(
                    label=(
                        "✅ Myanmar SRT "
                        "အောင်မြင်ပါပြီ"
                    ),
                    state="complete",
                )

            st.success(
                "မြန်မာ SRT ထုတ်ပြီးပါပြီ။"
            )

        except Exception as exc:

            st.error(
                "❌ SRT ထုတ်ရာမှာ Error ဖြစ်ပါတယ်"
            )

            st.code(
                str(exc)
            )


# ============================================================
# SRT RESULT
# ============================================================

if st.session_state.generated_srt:

    st.markdown(
        "### 📝 SRT Preview"
    )

    preview_text = (
        st.session_state.generated_srt
    )

    st.text_area(
        "Myanmar SRT",
        preview_text,
        height=300,
        key="srt_preview",
    )

    st.download_button(
        "⬇️ Myanmar SRT Download",
        data=preview_text.encode(
            "utf-8-sig"
        ),
        file_name=(
            "Myanmar_Subtitles.srt"
        ),
        mime=(
            "application/x-subrip"
        ),
        use_container_width=True,
    )


# ============================================================
# STEP 2
# ============================================================

st.markdown(
    """
<div class="card">
    <div class="card-title">
        ② SRT → မြန်မာ Voiceover
    </div>
    <div class="card-sub">
        Subtitle timing အတိုင်း voice ကို
        auto fit + volume normalize + sync လုပ်ပေးမယ်။
    </div>
</div>
""",
    unsafe_allow_html=True,
)

srt_file = st.file_uploader(
    "📝 SRT Upload",
    type=["srt"],
    key="voice_srt",
)

with st.form("voice_form"):

    profile_name = st.selectbox(
        "🎙️ Voice Character",
        list(
            VOICE_PROFILES.keys()
        ),
    )

    speed = st.slider(
        "⚡ Voice Speed",
        min_value=0.70,
        max_value=1.30,
        value=1.00,
        step=0.05,
    )

    output_name = st.text_input(
        "📁 Output Filename",
        value="Movie_Part_01.m4a",
    )

    test_voice = st.form_submit_button(
        "🔊 Test Voice",
        use_container_width=True,
    )

    generate_voice = st.form_submit_button(
        "🎙️ Voiceover Generate",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# TEST VOICE
# ============================================================

if test_voice:

    try:

        if srt_file is not None:

            raw = (
                srt_file
                .getvalue()
                .decode(
                    "utf-8-sig",
                    errors="replace",
                )
            )

            entries = parse_srt(
                raw
            )

        elif st.session_state.generated_srt:

            entries = parse_srt(
                st.session_state.generated_srt
            )

        else:

            entries = []

        if not entries:

            st.warning(
                "SRT အရင် Upload လုပ်ပါ "
                "သို့မဟုတ် Step 1 မှာ SRT ထုတ်ပါ။"
            )

        else:

            test_text = entries[0][
                "text"
            ]

            profile = (
                VOICE_PROFILES[
                    profile_name
                ]
            )

            temp_dir = tempfile.mkdtemp(
                prefix="voice_test_"
            )

            test_path = os.path.join(
                temp_dir,
                "test.mp3",
            )

            create_tts(
                test_text,
                test_path,
                profile["voice"],
                profile["rate"],
                profile["pitch"],
            )

            st.session_state.voice_preview = (
                test_path
            )

            st.success(
                "🔊 Test Voice အဆင်သင့်ပါပြီ"
            )

    except Exception as exc:

        st.error(
            "Test Voice Error"
        )

        st.code(
            str(exc)
        )


if st.session_state.voice_preview:

    st.audio(
        st.session_state.voice_preview
    )


# ============================================================
# GENERATE VOICEOVER
# ============================================================

if generate_voice:

    try:

        if srt_file is not None:

            raw_srt = (
                srt_file
                .getvalue()
                .decode(
                    "utf-8-sig",
                    errors="replace",
                )
            )

        elif st.session_state.generated_srt:

            raw_srt = (
                st.session_state.generated_srt
            )

        else:

            raw_srt = ""

        if not raw_srt.strip():

            st.error(
                "📝 SRT အရင် Upload လုပ်ပါ "
                "သို့မဟုတ် Step 1 မှာ SRT ထုတ်ပါ။"
            )

        else:

            entries = parse_srt(
                raw_srt
            )

            if not entries:
                raise RuntimeError(
                    "SRT format မမှန်ပါ။"
                )

            # ----------------------------------------
            # QUALITY CHECK
            # ----------------------------------------

            issues_before = (
                quality_check(
                    entries
                )
            )

            fixed = auto_fix_srt(
                entries
            )

            if fixed:

                st.info(
                    f"🛠️ Timestamp "
                    f"{fixed} ခု auto-fix လုပ်ပြီးပါပြီ။"
                )

            issues_after = (
                quality_check(
                    entries
                )
            )

            if issues_after:

                st.warning(
                    f"⚠️ SRT Quality Check: "
                    f"{len(issues_after)} issue(s)"
                )

                with st.expander(
                    "Quality Issues ကြည့်ရန်"
                ):

                    for issue in (
                        issues_after[:30]
                    ):
                        st.write(
                            "• "
                            + issue
                        )

            else:

                st.success(
                    "✅ SRT Quality Check — OK"
                )

            # ----------------------------------------
            # VOICE GENERATION
            # ----------------------------------------

            progress_bar = st.progress(
                0
            )

            status_box = st.empty()

            def voice_progress(
                value,
                text,
            ):
                progress_bar.progress(
                    min(
                        1.0,
                        max(
                            0.0,
                            float(value),
                        ),
                    )
                )

                status_box.info(
                    text
                )

            output_path = (
                build_voiceover(
                    entries,
                    profile_name,
                    speed,
                    voice_progress,
                )
            )

            final_name = safe_filename(
                output_name,
                "Myanmar_Voiceover.m4a",
            )

            if not final_name.lower().endswith(
                ".m4a"
            ):
                final_name += ".m4a"

            with open(
                output_path,
                "rb",
            ) as f:
                audio_bytes = f.read()

            st.session_state.generated_voice = {
                "bytes": audio_bytes,
                "name": final_name,
            }

            status_box.success(
                "✅ Myanmar Voiceover "
                "အောင်မြင်ပါပြီ"
            )

    except Exception as exc:

        st.error(
            "❌ Voiceover ထုတ်ရာမှာ Error ဖြစ်ပါတယ်"
        )

        st.code(
            str(exc)
        )


# ============================================================
# VOICE RESULT
# ============================================================

if st.session_state.generated_voice:

    result = (
        st.session_state.generated_voice
    )

    st.markdown(
        "### 🎧 Voiceover Preview"
    )

    st.audio(
        result["bytes"],
        format="audio/mp4",
    )

    st.download_button(
        "⬇️ Voiceover Download",
        data=result["bytes"],
        file_name=result["name"],
        mime="audio/mp4",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div style="
    text-align:center;
    color:#777f8c;
    font-size:12px;
    margin-top:35px;
">
    🎬 Myanmar Movie AI · AI Subtitle & Dubbing
</div>
""",
    unsafe_allow_html=True,
)
