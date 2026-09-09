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


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Myanmar Movie AI",
    page_icon="🎬",
    layout="wide",
)


# ============================================================
# FFMPEG
# ============================================================

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================
# VOICES
# ============================================================

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


# ============================================================
# GEMINI
# ============================================================

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
                rgba(0, 102, 255, 0.16),
                transparent 34%
            ),
            radial-gradient(
                circle at top right,
                rgba(255, 40, 70, 0.12),
                transparent 32%
            ),
            #050914;
    }

    .main-title {
        text-align: center;
        padding: 8px 0 18px 0;
    }

    .main-title h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 5px;
        background: linear-gradient(
            90deg,
            #32a8ff,
            #ffffff,
            #ff3b5c
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-title p {
        color: #aeb9cc;
        font-size: 0.95rem;
    }

    .section-card {
        border: 1px solid rgba(70, 130, 255, 0.25);
        border-radius: 18px;
        padding: 20px;
        margin: 12px 0 20px 0;
        background: rgba(10, 17, 32, 0.72);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.22);
    }

    .step-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin-bottom: 10px;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 48px;
    }

    div.stDownloadButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 46px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def run_cmd(args, timeout=1800):
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
    )


def get_secret(name):
    value = st.secrets.get(name, "")
    if value:
        return str(value).strip()

    return os.getenv(name, "").strip()


def gemini_client():
    key = get_secret("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return genai.Client(api_key=key)


def deepgram_key():
    key = get_secret("DEEPGRAM_API_KEY")

    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY မတွေ့ပါ။ Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return key


def clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


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

    name = name.strip("._ ")

    if name.lower().endswith(".m4a"):
        name = name[:-4]

    if not name:
        name = "myanmar_voiceover"

    return name + ".m4a"


# ============================================================
# MEDIA
# ============================================================

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
        raise RuntimeError(
            "Video/audio duration ကို ဖတ်မရပါ။"
        )

    return (
        int(m.group(1)) * 3600
        + int(m.group(2)) * 60
        + float(m.group(3))
    )


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
        r.returncode
        or not out.exists()
        or out.stat().st_size < 1000
    ):
        raise RuntimeError(
            "Video ထဲက audio ထုတ်မရပါ။\n"
            + (r.stderr or "")
        )


# ============================================================
# DEEPGRAM
# ============================================================

def deepgram_transcribe(audio_path):
    url = "https://api.deepgram.com/v1/listen"

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
        "Authorization": f"Token {deepgram_key()}",
        "Content-Type": "audio/wav",
    }

    data = audio_path.read_bytes()

    last = ""

    for attempt in range(3):
        try:
            r = requests.post(
                url,
                params=params,
                headers=headers,
                data=data,
                timeout=900,
            )

            if r.status_code == 200:
                obj = r.json()

                results = obj.get(
                    "results",
                    {},
                )

                channels = results.get(
                    "channels",
                    [],
                )

                if channels:
                    alternatives = channels[0].get(
                        "alternatives",
                        [],
                    )

                    if alternatives:
                        alt = alternatives[0]

                        words = alt.get(
                            "words",
                            [],
                        ) or []

                        if words:
                            return words

                utterances = results.get(
                    "utterances",
                    [],
                ) or []

                if utterances:
                    words = []

                    for u in utterances:
                        u_words = u.get(
                            "words",
                            [],
                        ) or []

                        if u_words:
                            words.extend(u_words)

                    if words:
                        return words

                    return utterances

                raise RuntimeError(
                    "Deepgram က dialogue/timestamp မပြန်ပေးပါ။"
                )

            last = (
                f"HTTP {r.status_code}: "
                f"{r.text[:500]}"
            )

            if r.status_code not in (
                429,
                500,
                502,
                503,
                504,
            ):
                break

        except Exception as e:
            last = str(e)

        time.sleep(
            2 ** attempt
            + random.random()
        )

    raise RuntimeError(
        "Deepgram STT မအောင်မြင်ပါ။\n"
        + last
    )


# ============================================================
# WORD HELPERS
# ============================================================

def get_word_text(word):
    return clean_text(
        word.get(
            "punctuated_word",
            word.get(
                "word",
                "",
            ),
        )
    )


def get_word_start(word):
    try:
        return float(
            word.get(
                "start",
                0,
            )
        )
    except Exception:
        return 0.0


def get_word_end(word):
    try:
        return float(
            word.get(
                "end",
                get_word_start(word),
            )
        )
    except Exception:
        return get_word_start(word)


def is_sentence_end(text):
    return str(text).rstrip().endswith(
        (
            ".",
            "!",
            "?",
            "。",
            "！",
            "？",
            "...",
            "…",
        )
    )


def is_strong_pause(previous_end, current_start):
    return (
        current_start - previous_end
        >= 0.65
    )


# ============================================================
# SMART SUBTITLE SEGMENTATION
# ============================================================

def build_subtitle_segments(raw_items):
    """
    Deepgram word timestamps ကို အသုံးပြုပြီး
    dialogue အရှည်ကြီးတွေကို subtitle အတိုအပိုင်းတွေခွဲသည်။

    Rules:
    - max characters ~ 42
    - max duration ~ 6 sec
    - long pause ရှိရင် ခွဲ
    - sentence punctuation ရှိရင် ခွဲ
    - စကားလုံးအလယ်မှာ မဖြတ်
    """

    words = []

    for item in raw_items:

        if "words" in item:
            source_words = item.get(
                "words",
                [],
            ) or []
        else:
            source_words = [item]

        for w in source_words:

            text = get_word_text(w)

            if not text:
                continue

            start = get_word_start(w)
            end = get_word_end(w)

            if end <= start:
                continue

            words.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                }
            )

    if not words:
        raise RuntimeError(
            "Dialogue word timestamp မတွေ့ပါ။"
        )

    words.sort(
        key=lambda x: x["start"]
    )

    segments = []

    current = []

    MAX_CHARS = 42
    MAX_DURATION = 6.0

    for word in words:

        if not current:
            current = [word]
            continue

        current_text = " ".join(
            x["text"]
            for x in current
        )

        proposed_text = (
            current_text
            + " "
            + word["text"]
        )

        current_start = current[0]["start"]
        current_end = current[-1]["end"]

        duration = (
            word["end"]
            - current_start
        )

        pause = (
            word["start"]
            - current_end
        )

        should_split = False

        # Sentence end
        if is_sentence_end(
            current[-1]["text"]
        ):
            should_split = True

        # Long pause
        if is_strong_pause(
            current_end,
            word["start"],
        ):
            should_split = True

        # Too long
        if (
            len(proposed_text)
            > MAX_CHARS
        ):
            should_split = True

        # Too long in time
        if duration > MAX_DURATION:
            should_split = True

        if should_split:

            if current:
                segments.append(
                    {
                        "start": current[0]["start"],
                        "end": current[-1]["end"],
                        "source": " ".join(
                            x["text"]
                            for x in current
                        ).strip(),
                    }
                )

            current = [word]

        else:
            current.append(word)

    if current:
        segments.append(
            {
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "source": " ".join(
                    x["text"]
                    for x in current
                ).strip(),
            }
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    cleaned = []

    for item in segments:

        start = max(
            0.0,
            float(item["start"]),
        )

        end = float(item["end"])

        text = clean_text(
            item["source"]
        )

        if not text:
            continue

        if end <= start:
            continue

        # Prevent accidental overlaps
        if cleaned:
            if start < cleaned[-1]["end"]:
                start = cleaned[-1]["end"]

        if end <= start:
            continue

        cleaned.append(
            {
                "start": start,
                "end": end,
                "source": text,
            }
        )

    if not cleaned:
        raise RuntimeError(
            "Valid subtitle segments မတွေ့ပါ။"
        )

    return cleaned


# ============================================================
# GEMINI JSON
# ============================================================

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

    a = text.find("[")
    b = text.rfind("]")

    if a >= 0 and b > a:
        return text[a:b + 1]

    return text


# ============================================================
# TRANSLATION
# ============================================================

def translate_batch(client, items):

    prompt = """
You are a professional movie subtitle translator.

Translate each source dialogue into natural,
conversational Burmese (Myanmar language).

IMPORTANT RULES:

1. Preserve the exact meaning.
2. Preserve names.
3. Preserve emotion and context.
4. Do not add explanations.
5. Do not summarize.
6. Do not merge different IDs.
7. Keep translations short enough for subtitle display.
8. Each result MUST keep the exact same ID.
9. Return ONLY valid JSON.
10. Do not use markdown.

The input already contains carefully timed,
short subtitle segments.

Return exactly this format:

[
  {
    "id": 1,
    "burmese": "မြန်မာစာ"
  }
]

INPUT:
""" + json.dumps(
        items,
        ensure_ascii=False,
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
                    not isinstance(
                        data,
                        list,
                    )
                    or len(data)
                    != len(items)
                ):
                    raise RuntimeError(
                        "Gemini translation result count မကိုက်ပါ။"
                    )

                by_id = {}

                for x in data:

                    try:
                        item_id = int(
                            x["id"]
                        )
                    except Exception:
                        continue

                    burmese = clean_text(
                        x.get(
                            "burmese",
                            "",
                        )
                    )

                    if burmese:
                        by_id[
                            item_id
                        ] = burmese

                expected_ids = [
                    int(x["id"])
                    for x in items
                ]

                if any(
                    item_id not in by_id
                    for item_id in expected_ids
                ):
                    raise RuntimeError(
                        "ဘာသာပြန်စာကြောင်းတချို့ မထွက်ပါ။"
                    )

                return {
                    item_id: by_id[item_id]
                    for item_id in expected_ids
                }

            except Exception as e:

                errors.append(
                    f"{model} "
                    f"attempt {attempt + 1}: "
                    f"{e}"
                )

                error_text = str(e).lower()

                if (
                    attempt == 0
                    and any(
                        x in error_text
                        for x in RETRY_WORDS
                    )
                ):
                    time.sleep(
                        3
                        + random.random() * 2
                    )
                else:
                    break

    raise RuntimeError(
        "Gemini translation မအောင်မြင်ပါ။\n"
        + "\n".join(
            errors[-8:]
        )
    )


# ============================================================
# BUILD FINAL SEGMENTS
# ============================================================

def build_segments(
    client,
    subtitle_segments,
    progress,
):

    result = []

    total = len(
        subtitle_segments
    )

    # Smaller batches because subtitle
    # segments are now more numerous.
    batch_size = 20

    for pos in range(
        0,
        total,
        batch_size,
    ):

        batch = subtitle_segments[
            pos:
            pos + batch_size
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

        progress(
            0.20
            + 0.50
            * (
                pos
                / max(total, 1)
            ),
            (
                "Gemini ဘာသာပြန်နေသည်... "
                f"{min(pos + len(batch), total)}"
                f"/{total}"
            ),
        )

        translated = translate_batch(
            client,
            payload,
        )

        for i, item in enumerate(
            batch,
            1,
        ):

            burmese = clean_text(
                translated[i]
            )

            if not burmese:
                continue

            result.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "burmese": burmese,
                }
            )

    if not result:
        raise RuntimeError(
            "မြန်မာဘာသာပြန် subtitle မထွက်ပါ။"
        )

    return result


# ============================================================
# SRT
# ============================================================

def srt_time(seconds):

    seconds = max(
        0.0,
        float(seconds),
    )

    ms = int(
        round(
            seconds * 1000
        )
    )

    h, rem = divmod(
        ms,
        3600000,
    )

    m, rem = divmod(
        rem,
        60000,
    )

    s, ms = divmod(
        rem,
        1000,
    )

    return (
        f"{h:02d}:"
        f"{m:02d}:"
        f"{s:02d},"
        f"{ms:03d}"
    )


def make_srt(segments):

    blocks = []

    for i, segment in enumerate(
        segments,
        1,
    ):

        blocks.append(
            f"{i}\n"
            f"{srt_time(segment['start'])}"
            f" --> "
            f"{srt_time(segment['end'])}\n"
            f"{segment['burmese']}\n"
        )

    return "\n".join(
        blocks
    )


# ============================================================
# SRT PARSER
# ============================================================

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

    out = []

    for block in blocks:

        lines = [
            x.strip("\ufeff")
            for x in block.split("\n")
        ]

        if len(lines) < 3:
            continue

        time_line = next(
            (
                x
                for x in lines
                if "-->" in x
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

            h, minute, sec = (
                value
                .replace(",", ".")
                .split(":")
            )

            return (
                int(h) * 3600
                + int(minute) * 60
                + float(sec)
            )

        index = lines.index(
            time_line
        )

        text_lines = [
            x.strip()
            for x in lines[index + 1:]
            if x.strip()
        ]

        text_value = clean_text(
            " ".join(text_lines)
        )

        if not text_value:
            continue

        out.append(
            {
                "start": parse_time(
                    match.group(1)
                ),
                "end": parse_time(
                    match.group(2)
                ),
                "burmese": text_value,
            }
        )

    out.sort(
        key=lambda x: x["start"]
    )

    cleaned = []

    for item in out:

        start = float(
            item["start"]
        )

        end = float(
            item["end"]
        )

        if end <= start:
            continue

        # Auto-fix overlap
        if cleaned:
            if start < cleaned[-1]["end"]:
                start = cleaned[-1]["end"]

        if end <= start:
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
            "SRT ထဲမှာ valid subtitle မတွေ့ပါ။"
        )

    return cleaned


# ============================================================
# EDGE TTS
# ============================================================

async def edge_tts_save(
    text,
    voice,
    rate,
    pitch,
    out,
):

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=f"{rate:+d}%",
        pitch=f"{pitch:+d}Hz",
    )

    await communicate.save(
        str(out)
    )


def make_tts(
    text,
    voice,
    style,
    out,
):

    cfg = VOICE_STYLES[
        style
    ]

    errors = []

    for attempt in range(3):

        try:

            if out.exists():
                out.unlink()

            asyncio.run(
                edge_tts_save(
                    text,
                    voice,
                    cfg["rate"],
                    cfg["pitch"],
                    out,
                )
            )

            if (
                out.exists()
                and out.stat().st_size
                > 1000
            ):
                return

            raise RuntimeError(
                "TTS file အလွတ်ဖြစ်နေပါသည်။"
            )

        except Exception as e:

            errors.append(
                str(e)
            )

            time.sleep(
                2 + attempt
            )

    raise RuntimeError(
        "Burmese TTS မအောင်မြင်ပါ။\n"
        + "\n".join(
            errors[-3:]
        )
    )


# ============================================================
# AUDIO SPEED
# ============================================================

def atempo_chain(speed):

    speed = max(
        0.25,
        min(
            float(speed),
            4.0,
        ),
    )

    parts = []

    while speed > 2.0:
        parts.append(
            "atempo=2.0"
        )
        speed /= 2.0

    while speed < 0.5:
        parts.append(
            "atempo=0.5"
        )
        speed /= 0.5

    parts.append(
        f"atempo={speed:.6f}"
    )

    return ",".join(
        parts
    )


# ============================================================
# FIT TTS TO SRT SLOT
# ============================================================

def fit_clip(
    src,
    out,
    slot,
    user_speed,
):

    raw_duration = ffprobe_duration(
        src
    )

    desired_speed = max(
        0.25,
        min(
            float(user_speed),
            2.0,
        ),
    )

    final_speed = (
        raw_duration
        / max(slot, 0.05)
    ) * desired_speed

    final_speed = max(
        0.25,
        min(
            final_speed,
            4.0,
        ),
    )

    filter_audio = (
        atempo_chain(
            final_speed
        )
        + ",apad,"
        + "atrim=duration="
        + f"{slot:.3f}"
    )

    r = run_cmd(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
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
            str(out),
        ],
        180,
    )

    if (
        r.returncode
        or not out.exists()
        or out.stat().st_size < 1000
    ):
        raise RuntimeError(
            "Voiceover timing ပြင်မရပါ။\n"
            + (r.stderr or "")
        )


# ============================================================
# BUILD VOICEOVER
# ============================================================

def build_voiceover(
    segments,
    voice,
    style,
    speed,
    work,
    progress,
):

    clips = []

    total = len(
        segments
    )

    for i, segment in enumerate(
        segments,
        1,
    ):

        start = max(
            0.0,
            float(
                segment["start"]
            ),
        )

        end = max(
            start + 0.05,
            float(
                segment["end"]
            ),
        )

        slot = end - start

        raw = (
            work
            / f"tts_{i:04d}.mp3"
        )

        fitted = (
            work
            / f"clip_{i:04d}.m4a"
        )

        progress(
            0.05
            + 0.70
            * (
                (i - 1)
                / max(total, 1)
            ),
            (
                f"Voice {i}/{total} "
                "ထုတ်နေသည်..."
            ),
        )

        make_tts(
            segment["burmese"],
            voice,
            style,
            raw,
        )

        fit_clip(
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
            "Voiceover segment မရှိပါ။"
        )

    out = (
        work
        / "burmese_voiceover.m4a"
    )

    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    for _, clip in clips:
        cmd += [
            "-i",
            str(clip),
        ]

    filters = []
    labels = []

    for i, (
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

        label = f"v{i}"

        filters.append(
            f"[{i}:a]"
            f"adelay={delay_ms}:all=1,"
            f"aresample=48000"
            f"[{label}]"
        )

        labels.append(
            f"[{label}]"
        )

    # IMPORTANT:
    # Normalize final voiceover louder,
    # then limiter prevents clipping.
    mix_filter = (
        "".join(labels)
        + f"amix="
        f"inputs={len(labels)}:"
        f"duration=longest:"
        f"dropout_transition=0,"
        "loudnorm="
        "I=-14:"
        "TP=-1.5:"
        "LRA=7,"
        "alimiter="
        "limit=0.95"
        "[out]"
    )

    filters.append(
        mix_filter
    )

    cmd += [
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
        str(out),
    ]

    r = run_cmd(
        cmd,
        1800,
    )

    if (
        r.returncode
        or not out.exists()
        or out.stat().st_size < 5000
    ):
        raise RuntimeError(
            "Voiceover file မထုတ်နိုင်ပါ။\n"
            + (r.stderr or "")
        )

    # Final decode validation
    check = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(out),
            "-f",
            "null",
            "-",
        ],
        300,
    )

    if check.returncode:
        raise RuntimeError(
            "Voiceover audio validation မအောင်မြင်ပါ။\n"
            + (check.stderr or "")
        )

    progress(
        1.0,
        "Voiceover ပြီးပါပြီ",
    )

    return out


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        <h1>🎬 Myanmar Movie AI</h1>
        <p>
            Video → 🇲🇲 Natural Myanmar Subtitle
            → 🎙️ Burmese Voiceover
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STEP 1
# ============================================================

st.markdown(
    """
    <div class="section-card">
        <div class="step-title">
            1️⃣ Video → မြန်မာ SRT
        </div>
    </div>
    """,
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
    key="video",
)

make_srt_button = st.button(
    "📝 မြန်မာ SRT ထုတ်မယ်",
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

        with tempfile.TemporaryDirectory() as td:

            work = Path(td)

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
            bar = st.progress(
                0.0
            )

            # ---------------------------------------------
            # Extract
            # ---------------------------------------------

            status.info(
                "Video audio ထုတ်နေသည်..."
            )

            extract_audio(
                video,
                audio,
            )

            bar.progress(
                0.12
            )

            # ---------------------------------------------
            # Deepgram
            # ---------------------------------------------

            status.info(
                "AI က စကားလုံးတစ်လုံးချင်း "
                "timestamp ရယူနေသည်..."
            )

            raw_transcript = (
                deepgram_transcribe(
                    audio
                )
            )

            bar.progress(
                0.28
            )

            # ---------------------------------------------
            # Smart Segmentation
            # ---------------------------------------------

            status.info(
                "Dialogue တွေကို "
                "စာတန်းထိုးဖို့ သင့်တော်တဲ့အပိုင်းတွေ "
                "အလိုအလျောက်ခွဲနေသည်..."
            )

            subtitle_segments = (
                build_subtitle_segments(
                    raw_transcript
                )
            )

            bar.progress(
                0.38
            )

            # ---------------------------------------------
            # Translation
            # ---------------------------------------------

            status.info(
                "Gemini က မြန်မာလို "
                "သဘာဝကျကျ ဘာသာပြန်နေသည်..."
            )

            segments = build_segments(
                gemini_client(),
                subtitle_segments,
                lambda p, t: (
                    bar.progress(
                        min(
                            p,
                            0.92,
                        )
                    ),
                    status.info(t),
                ),
            )

            # ---------------------------------------------
            # SRT
            # ---------------------------------------------

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

            bar.progress(
                1.0
            )

            status.success(
                "SRT ပြီးပါပြီ — "
                f"{len(segments)} subtitle lines"
            )

    except Exception as e:

        st.error(
            "SRT ထုတ်ရာမှာ အမှားဖြစ်ပါတယ်။"
        )

        st.exception(e)


# ============================================================
# SRT PREVIEW
# ============================================================

if st.session_state.get(
    "srt_text"
):

    st.subheader(
        "📄 SRT Preview"
    )

    st.text_area(
        "မြန်မာ SRT",
        st.session_state[
            "srt_text"
        ],
        height=320,
    )

    st.download_button(
        "⬇️ Download Myanmar SRT",
        st.session_state[
            "srt_text"
        ].encode(
            "utf-8-sig"
        ),
        st.session_state.get(
            "srt_name",
            "myanmar.srt",
        ),
        "application/x-subrip",
        use_container_width=True,
    )


# ============================================================
# STEP 2
# ============================================================

st.markdown(
    """
    <div class="section-card">
        <div class="step-title">
            2️⃣ SRT → မြန်မာ Voiceover
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

srt_file = st.file_uploader(
    "📄 SRT တင်ပါ",
    type=["srt"],
    key="srt_upload",
)


# ============================================================
# VOICE FORM
# ============================================================

with st.form(
    "voice_generation_form"
):

    voice_name = st.selectbox(
        "🎙️ Voice",
        list(VOICES),
    )

    style = st.selectbox(
        "🎭 Voice Style",
        list(VOICE_STYLES),
    )

    speed = st.slider(
        "⚡ Speed",
        0.70,
        1.30,
        1.00,
        0.05,
    )

    output_filename_input = st.text_input(
        "📁 Voice File Name",
        value="myanmar_voiceover",
        help="ဥပမာ - Movie_Part_01",
    )

    make_voice_button = st.form_submit_button(
        "🎙️ Voiceover စတင်ထုတ်မယ်",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# VOICEOVER PROCESS
# ============================================================

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
            "(သို့) အဆင့် ၁ မှာ SRT အရင်ထုတ်ပါ။"
        )

        st.stop()

    # Snapshot settings before processing.
    # This prevents accidental widget changes
    # from affecting the running job.

    job_voice = VOICES[
        voice_name
    ]

    job_style = style

    job_speed = float(
        speed
    )

    job_filename = safe_filename(
        output_filename_input
    )

    try:

        segments = parse_srt(
            source_srt
        )

        with tempfile.TemporaryDirectory() as td:

            work = Path(td)

            status = st.empty()

            bar = st.progress(
                0.0
            )

            status.info(
                "SRT timing စစ်နေသည်... "
                f"{len(segments)} lines"
            )

            voiceover = build_voiceover(
                segments,
                job_voice,
                job_style,
                job_speed,
                work,
                lambda p, t: (
                    bar.progress(
                        min(
                            p,
                            1.0,
                        )
                    ),
                    status.info(t),
                ),
            )

            data = (
                voiceover.read_bytes()
            )

            st.session_state[
                "voice_bytes"
            ] = data

            st.session_state[
                "voice_name"
            ] = job_filename

            st.session_state[
                "voice_mime"
            ] = "audio/mp4"

            st.session_state[
                "voice_segments"
            ] = segments

            bar.progress(
                1.0
            )

            status.success(
                "Voiceover ပြီးပါပြီ"
            )

    except Exception as e:

        st.error(
            "Voiceover ထုတ်ရာမှာ အမှားဖြစ်ပါတယ်။"
        )

        st.exception(e)


# ============================================================
# VOICEOVER RESULT
# ============================================================

if st.session_state.get(
    "voice_bytes"
):

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
