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

# ============================================================
# STEP 3 — MANUAL VIDEO EDIT STUDIO
# ============================================================

def has_audio_stream(path):
    r = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(path),
        ],
        120,
    )

    return bool(
        re.search(
            r"Stream #.*Audio:",
            r.stderr or "",
            re.I,
        )
    )


def ffmpeg_escape_path(path):
    """
    Escape a filesystem path for FFmpeg filter arguments.
    """
    value = str(path).replace("\\", "/")
    value = value.replace(":", "\\:")
    value = value.replace("'", "\\'")
    value = value.replace(",", "\\,")
    value = value.replace("[", "\\[")
    value = value.replace("]", "\\]")
    return value


def ass_escape(text):
    text = str(text or "")
    text = text.replace("\\", r"\\")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    return text


def ass_time(seconds):
    seconds = max(0.0, float(seconds))

    h = int(seconds // 3600)
    seconds -= h * 3600

    m = int(seconds // 60)
    seconds -= m * 60

    s = int(seconds)
    cs = int(round((seconds - s) * 100))

    if cs >= 100:
        s += 1
        cs = 0

    if s >= 60:
        s = 0
        m += 1

    if m >= 60:
        m = 0
        h += 1

    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_color(hex_color):
    """
    #FFFFFF -> &H00FFFFFF
    """
    value = str(hex_color or "#FFFFFF").strip()

    if not value.startswith("#"):
        value = "#" + value

    value = value[1:]

    if len(value) != 6:
        value = "FFFFFF"

    rr = value[0:2]
    gg = value[2:4]
    bb = value[4:6]

    return f"&H00{bb}{gg}{rr}"


def make_ass_subtitles(
    segments,
    out_path,
    font_size=44,
    color="#FFFFFF",
    position="အောက်",
    outline=3,
):
    """
    Convert SRT-like Burmese subtitle segments into ASS.
    """

    if position == "အပေါ်":
        alignment = 8
        margin_v = 55

    elif position == "အလယ်":
        alignment = 5
        margin_v = 20

    else:
        alignment = 2
        margin_v = 55

    primary = ass_color(color)

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, "
        "PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Noto Sans Myanmar,"
        f"{int(font_size)},"
        f"{primary},"
        f"{primary},"
        f"&H00000000,"
        f"&H80000000,"
        f"0,0,0,0,100,100,0,0,1,"
        f"{int(outline)},2,"
        f"{alignment},40,40,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, "
        "Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for item in segments:

        start = float(item["start"])
        end = float(item["end"])

        text = ass_escape(
            item.get(
                "burmese",
                "",
            )
        )

        if not text:
            continue

        # Long Burmese lines are automatically wrapped.
        if len(text) > 28:
            words = text.split(" ")

            result_lines = []
            current = ""

            for word in words:

                proposed = (
                    word
                    if not current
                    else current + " " + word
                )

                if len(proposed) > 28:
                    if current:
                        result_lines.append(
                            current
                        )

                    current = word

                else:
                    current = proposed

            if current:
                result_lines.append(
                    current
                )

            text = r"\N".join(
                result_lines
            )

        lines.append(
            "Dialogue: 0,"
            f"{ass_time(start)},"
            f"{ass_time(end)},"
            "Default,,0,0,0,,"
            f"{text}"
        )

    out_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_ratio(value):
    if value == "9:16":
        return 1080, 1920

    if value == "16:9":
        return 1920, 1080

    if value == "1:1":
        return 1080, 1080

    if value == "4:5":
        return 1080, 1350

    return None


def build_edit_video(
    video_path,
    output_path,
    trim_start,
    trim_end,
    ratio,
    flip_h,
    flip_v,
    zoom,
    pos_x,
    pos_y,
    blur_background,
    mask_type,
    srt_segments,
    subtitle_size,
    subtitle_color,
    subtitle_position,
    subtitle_outline,
    voice_path,
    music_path,
    original_audio,
    original_volume,
    voice_volume,
    music_volume,
    quality,
):

    work = output_path.parent

    # --------------------------------------------------------
    # Validate duration
    # --------------------------------------------------------

    source_duration = ffprobe_duration(
        video_path
    )

    trim_start = max(
        0.0,
        min(
            float(trim_start),
            source_duration,
        ),
    )

    trim_end = max(
        trim_start + 0.05,
        min(
            float(trim_end),
            source_duration,
        ),
    )

    final_duration = (
        trim_end - trim_start
    )

    if final_duration <= 0:
        raise RuntimeError(
            "Trim duration မမှန်ပါ။"
        )

    # --------------------------------------------------------
    # Detect original audio
    # --------------------------------------------------------

    source_has_audio = has_audio_stream(
        video_path
    )

    use_original = (
        original_audio
        and source_has_audio
    )

    # --------------------------------------------------------
    # Prepare ASS
    # --------------------------------------------------------

    ass_path = (
        work
        / "subtitles.ass"
    )

    shifted_srt = []

    for item in srt_segments:

        start = float(
            item["start"]
        ) - trim_start

        end = float(
            item["end"]
        ) - trim_start

        if end <= 0:
            continue

        start = max(
            0.0,
            start,
        )

        end = min(
            final_duration,
            end,
        )

        if end <= start:
            continue

        shifted_srt.append(
            {
                "start": start,
                "end": end,
                "burmese": item[
                    "burmese"
                ],
            }
        )

    if shifted_srt:
        make_ass_subtitles(
            shifted_srt,
            ass_path,
            subtitle_size,
            subtitle_color,
            subtitle_position,
            subtitle_outline,
        )

    # --------------------------------------------------------
    # Video filter
    # --------------------------------------------------------

    video_chain = [
        f"trim=start={trim_start:.3f}:"
        f"end={trim_end:.3f}",
        "setpts=PTS-STARTPTS",
    ]

    # Flip
    if flip_h:
        video_chain.append(
            "hflip"
        )

    if flip_v:
        video_chain.append(
            "vflip"
        )

    # Zoom
    zoom = max(
        1.0,
        min(
            float(zoom),
            2.5,
        ),
    )

    if zoom > 1.001:
        video_chain.append(
            f"scale="
            f"iw*{zoom:.4f}:"
            f"ih*{zoom:.4f}:"
            "flags=lanczos"
        )

    # --------------------------------------------------------
    # Ratio / Canvas
    # --------------------------------------------------------

    target = parse_ratio(
        ratio
    )

    if target:

        target_w, target_h = target

        # Background blur
        if blur_background:

            bg_chain = [
                f"scale={target_w}:{target_h}:"
                "force_original_aspect_ratio=increase",
                f"crop={target_w}:{target_h}",
                "boxblur=20:10",
            ]

            fg_chain = [
                f"scale={target_w}:{target_h}:"
                "force_original_aspect_ratio=decrease",
            ]

            # Position
            x_expr = (
                f"(W-w)/2+{int(pos_x)}"
            )

            y_expr = (
                f"(H-h)/2+{int(pos_y)}"
            )

            fg_chain.append(
                "format=rgba"
            )

            filter_complex_video = (
                "[0:v]"
                + ",".join(
                    video_chain
                    + bg_chain
                )
                + "[bg];"
                "[0:v]"
                + ",".join(
                    video_chain
                    + fg_chain
                )
                + "[fg];"
                "[bg][fg]"
                f"overlay="
                f"x={x_expr}:"
                f"y={y_expr}"
                "[base]"
            )

        else:

            canvas = (
                f"scale={target_w}:{target_h}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:"
                f"(ow-iw)/2+{int(pos_x)}:"
                f"(oh-ih)/2+{int(pos_y)}:"
                "black"
            )

            filter_complex_video = (
                "[0:v]"
                + ","
                .join(
                    video_chain
                    + [canvas]
                )
                + "[base]"
            )

    else:

        # Original ratio
        filter_complex_video = (
            "[0:v]"
            + ","
            .join(
                video_chain
            )
            + "[base]"
        )

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    if mask_type == "စက်ဝိုင်း":

        # Safe circular mask.
        filter_complex_video += (
            ";[base]"
            "format=rgba,"
            "geq="
            "r='r(X,Y)':"
            "g='g(X,Y)':"
            "b='b(X,Y)':"
            "a='if("
            "lte("
            "pow(X-W/2,2)+"
            "pow(Y-H/2,2),"
            "pow(min(W,H)/2,2)"
            "),255,0)'"
            "[masked]"
        )

        # Put black behind transparent area.
        if target:
            tw, th = target

            filter_complex_video += (
                f";color=c=black:"
                f"s={tw}x{th}:"
                f"d={final_duration:.3f}"
                "[maskbg];"
                "[maskbg][masked]"
                "overlay=shortest=1"
                "[base2]"
            )

            current_video_label = (
                "[base2]"
            )

        else:

            filter_complex_video += (
                ";color=c=black:"
                f"s=1920x1080:"
                f"d={final_duration:.3f}"
                "[maskbg];"
                "[maskbg][masked]"
                "overlay=shortest=1"
                "[base2]"
            )

            current_video_label = (
                "[base2]"
            )

    else:
        current_video_label = "[base]"

    # --------------------------------------------------------
    # Rectangle mask
    # --------------------------------------------------------

    if mask_type == "Rectangle":

        filter_complex_video += (
            f";{current_video_label}"
            "drawbox="
            "x=0:y=0:"
            "w=iw:h=ih:"
            "color=black@0:"
            "t=fill"
            "[masked_rect]"
        )

        current_video_label = (
            "[masked_rect]"
        )

    # --------------------------------------------------------
    # Subtitles
    # --------------------------------------------------------

    if shifted_srt:

        ass_file = ffmpeg_escape_path(
            ass_path
        )

        filter_complex_video += (
            f";{current_video_label}"
            f"subtitles='{ass_file}'"
            "[vout]"
        )

    else:

        filter_complex_video += (
            f";{current_video_label}"
            "[vout]"
        )

    # --------------------------------------------------------
    # Audio filters
    # --------------------------------------------------------

    audio_filters = []
    audio_labels = []

    input_count = 1

    # Original audio
    if use_original:

        audio_filters.append(
            "[0:a]"
            f"atrim=start={trim_start:.3f}:"
            f"end={trim_end:.3f},"
            "asetpts=PTS-STARTPTS,"
            f"volume={float(original_volume):.3f}"
            "[orig]"
        )

        audio_labels.append(
            "[orig]"
        )

    # Voiceover
    if voice_path:

        voice_index = input_count
        input_count += 1

        audio_filters.append(
            f"[{voice_index}:a]"
            f"atrim=duration={final_duration:.3f},"
            "asetpts=PTS-STARTPTS,"
            f"volume={float(voice_volume):.3f},"
            "aresample=48000"
            "[voice]"
        )

        audio_labels.append(
            "[voice]"
        )

    # Background music
    if music_path:

        music_index = input_count
        input_count += 1

        audio_filters.append(
            f"[{music_index}:a]"
            f"atrim=duration={final_duration:.3f},"
            "asetpts=PTS-STARTPTS,"
            f"volume={float(music_volume):.3f},"
            "aresample=48000,"
            "apad,"
            f"atrim=duration={final_duration:.3f}"
            "[music]"
        )

        audio_labels.append(
            "[music]"
        )

    # --------------------------------------------------------
    # Final audio mix
    # --------------------------------------------------------

    if audio_labels:

        if len(audio_labels) == 1:

            audio_filters.append(
                audio_labels[0]
                + "loudnorm="
                "I=-14:"
                "TP=-1.5:"
                "LRA=7,"
                "alimiter=limit=0.95"
                "[aout]"
            )

        else:

            audio_filters.append(
                "".join(
                    audio_labels
                )
                + f"amix="
                f"inputs={len(audio_labels)}:"
                "duration=longest:"
                "dropout_transition=0,"
                "loudnorm="
                "I=-14:"
                "TP=-1.5:"
                "LRA=7,"
                "alimiter=limit=0.95"
                "[aout]"
            )

    # --------------------------------------------------------
    # FFmpeg command
    # --------------------------------------------------------

    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
    ]

    if voice_path:
        cmd += [
            "-i",
            str(voice_path),
        ]

    if music_path:
        cmd += [
            "-stream_loop",
            "-1",
            "-i",
            str(music_path),
        ]

    all_filters = [
        filter_complex_video
    ]

    if audio_filters:
        all_filters.append(
            ";".join(audio_filters)
        )

    cmd += [
        "-filter_complex",
        ";".join(
            all_filters
        ),
        "-map",
        "[vout]",
    ]

    if audio_labels:

        cmd += [
            "-map",
            "[aout]",
        ]

    else:

        cmd += [
            "-an",
        ]

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    crf_map = {
        "အရည်အသွေးမြင့်": "18",
        "ပုံမှန်": "21",
        "ဖိုင်သေး": "26",
    }

    crf = crf_map.get(
        quality,
        "21",
    )

    cmd += [
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
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "-t",
        f"{final_duration:.3f}",
        str(output_path),
    ]

    r = run_cmd(
        cmd,
        timeout=3600,
    )

    if (
        r.returncode
        or not output_path.exists()
        or output_path.stat().st_size < 10000
    ):
        raise RuntimeError(
            "Final video render မအောင်မြင်ပါ။\n\n"
            + (r.stderr or "Unknown FFmpeg error")
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    check = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(output_path),
            "-f",
            "null",
            "-",
        ],
        600,
    )

    if check.returncode:
        raise RuntimeError(
            "Final MP4 validation မအောင်မြင်ပါ။\n"
            + (check.stderr or "")
        )

    return output_path


# ============================================================
# STEP 3 UI
# ============================================================

st.markdown(
    """
    <div class="section-card">
        <div class="step-title">
            3️⃣ Manual Video Edit Studio
        </div>
        <div style="color:#aeb9cc;">
            CapCut မသွားဘဲ ဒီနေရာကနေ Video + Voiceover
            + SRT + Background Music ကို တစ်ခါတည်း
            ပြင်ပြီး Final MP4 ထုတ်နိုင်ပါတယ်။
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


edit_video_file = st.file_uploader(
    "🎥 Edit လုပ်မယ့် Video",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm",
    ],
    key="edit_video_upload",
)


# ------------------------------------------------------------
# Voiceover source
# ------------------------------------------------------------

st.subheader(
    "🎙️ Voiceover"
)

voice_source = st.radio(
    "Voiceover Source",
    [
        "Step 2 မှာထုတ်ထားတာသုံးမယ်",
        "Voiceover ဖိုင်တင်မယ်",
        "မထည့်ပါ",
    ],
    horizontal=True,
    key="edit_voice_source",
)

edit_voice_file = None

if voice_source == "Voiceover ဖိုင်တင်မယ်":

    edit_voice_file = st.file_uploader(
        "🔊 Voiceover",
        type=[
            "m4a",
            "mp3",
            "wav",
            "aac",
            "mp4",
        ],
        key="edit_voice_upload",
    )

elif voice_source == "Step 2 မှာထုတ်ထားတာသုံးမယ်":

    if st.session_state.get(
        "voice_bytes"
    ):

        st.success(
            "✅ Step 2 Voiceover ကို အသုံးပြုမည်"
        )

    else:

        st.warning(
            "Step 2 Voiceover မရှိသေးပါ။"
        )


# ------------------------------------------------------------
# SRT source
# ------------------------------------------------------------

st.subheader(
    "📝 Subtitle"
)

srt_source = st.radio(
    "SRT Source",
    [
        "Step 1 မှာထုတ်ထားတာသုံးမယ်",
        "SRT ဖိုင်တင်မယ်",
        "Subtitle မထည့်ပါ",
    ],
    horizontal=True,
    key="edit_srt_source",
)

edit_srt_file = None

if srt_source == "SRT ဖိုင်တင်မယ်":

    edit_srt_file = st.file_uploader(
        "📄 SRT",
        type=["srt"],
        key="edit_srt_upload",
    )

elif srt_source == "Step 1 မှာထုတ်ထားတာသုံးမယ်":

    if st.session_state.get(
        "srt_text"
    ):

        st.success(
            "✅ Step 1 SRT ကို အသုံးပြုမည်"
        )

    else:

        st.warning(
            "Step 1 SRT မရှိသေးပါ။"
        )


# ------------------------------------------------------------
# Canvas / Transform
# ------------------------------------------------------------

st.subheader(
    "🖼️ Canvas & Transform"
)

c1, c2 = st.columns(2)

with c1:

    edit_ratio = st.selectbox(
        "📐 Ratio",
        [
            "Original",
            "9:16",
            "16:9",
            "1:1",
            "4:5",
        ],
        key="edit_ratio",
    )

    flip_h = st.checkbox(
        "↔️ Horizontal Flip",
        key="edit_flip_h",
    )

    flip_v = st.checkbox(
        "↕️ Vertical Flip",
        key="edit_flip_v",
    )

    blur_background = st.checkbox(
        "🌫️ Background Blur",
        key="edit_blur_background",
    )

with c2:

    zoom = st.slider(
        "🔍 Zoom",
        1.00,
        2.50,
        1.00,
        0.05,
        key="edit_zoom",
    )

    pos_x = st.slider(
        "↔️ Position X",
        -500,
        500,
        0,
        10,
        key="edit_pos_x",
    )

    pos_y = st.slider(
        "↕️ Position Y",
        -500,
        500,
        0,
        10,
        key="edit_pos_y",
    )


mask_type = st.selectbox(
    "🎭 Mask",
    [
        "မရှိ",
        "စက်ဝိုင်း",
        "Rectangle",
    ],
    key="edit_mask",
)


# ------------------------------------------------------------
# Subtitle style
# ------------------------------------------------------------

if srt_source != "Subtitle မထည့်ပါ":

    st.subheader(
        "✍️ Subtitle Style"
    )

    sc1, sc2 = st.columns(2)

    with sc1:

        subtitle_size = st.slider(
            "စာလုံးအရွယ်",
            24,
            72,
            44,
            2,
            key="edit_subtitle_size",
        )

        subtitle_position = st.selectbox(
            "နေရာ",
            [
                "အောက်",
                "အလယ်",
                "အပေါ်",
            ],
            key="edit_subtitle_position",
        )

    with sc2:

        subtitle_color = st.color_picker(
            "စာလုံးအရောင်",
            "#FFFFFF",
            key="edit_subtitle_color",
        )

        subtitle_outline = st.slider(
            "Outline",
            0,
            8,
            3,
            1,
            key="edit_subtitle_outline",
        )

else:

    subtitle_size = 44
    subtitle_position = "အောက်"
    subtitle_color = "#FFFFFF"
    subtitle_outline = 3


# ------------------------------------------------------------
# Audio
# ------------------------------------------------------------

st.subheader(
    "🔊 Audio"
)

ac1, ac2 = st.columns(2)

with ac1:

    original_audio = st.checkbox(
        "🎬 Original Audio ထည့်မယ်",
        value=False,
        key="edit_original_audio",
    )

    original_volume = st.slider(
        "Original Volume",
        0.0,
        2.0,
        0.25,
        0.05,
        key="edit_original_volume",
    )

    voice_volume = st.slider(
        "🎙️ Voiceover Volume",
        0.0,
        2.0,
        1.0,
        0.05,
        key="edit_voice_volume",
    )

with ac2:

    music_volume = st.slider(
        "🎵 BGM Volume",
        0.0,
        1.0,
        0.15,
        0.05,
        key="edit_music_volume",
    )

    music_file = st.file_uploader(
        "🎵 Background Music",
        type=[
            "mp3",
            "m4a",
            "wav",
            "aac",
        ],
        key="edit_music_upload",
    )


# ------------------------------------------------------------
# Trim
# ------------------------------------------------------------

st.subheader(
    "✂️ Trim"
)

edit_duration = 0.0

if edit_video_file:

    # Save temporarily only for duration check.
    try:

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False,
        ) as temp_video:

            temp_video.write(
                edit_video_file.getbuffer()
            )

            temp_video_path = Path(
                temp_video.name
            )

        try:

            edit_duration = ffprobe_duration(
                temp_video_path
            )

        finally:

            try:
                temp_video_path.unlink()
            except Exception:
                pass

    except Exception:

        edit_duration = 0.0


if edit_duration > 0:

    trim_start = st.number_input(
        "Start (seconds)",
        min_value=0.0,
        max_value=float(
            max(
                0.0,
                edit_duration - 0.05,
            )
        ),
        value=0.0,
        step=0.1,
        key="edit_trim_start",
    )

    trim_end = st.number_input(
        "End (seconds)",
        min_value=0.05,
        max_value=float(
            edit_duration
        ),
        value=float(
            edit_duration
        ),
        step=0.1,
        key="edit_trim_end",
    )

else:

    trim_start = 0.0
    trim_end = 0.0

    st.info(
        "Video တင်ပြီးရင် Trim controls ပေါ်လာပါမယ်။"
    )


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

st.subheader(
    "💾 Output"
)

quality = st.selectbox(
    "Quality",
    [
        "အရည်အသွေးမြင့်",
        "ပုံမှန်",
        "ဖိုင်သေး",
    ],
    index=1,
    key="edit_quality",
)

final_filename = st.text_input(
    "📁 Final Video File Name",
    value="Myanmar_Movie_Final",
    key="edit_final_filename",
)

render_button = st.button(
    "🎬 FINAL VIDEO ထုတ်မယ်",
    type="primary",
    use_container_width=True,
    key="render_final_video",
)


# ============================================================
# STEP 3 PROCESS
# ============================================================

if render_button:

    if not edit_video_file:

        st.error(
            "🎥 Video တစ်ခုအရင်တင်ပါ။"
        )

        st.stop()

    if edit_duration <= 0:

        st.error(
            "Video duration ကို မဖတ်နိုင်ပါ။"
        )

        st.stop()

    if trim_end <= trim_start:

        st.error(
            "❌ Trim Start / End မမှန်ပါ။"
        )

        st.stop()

    # --------------------------------------------------------
    # Get SRT
    # --------------------------------------------------------

    edit_srt_text = ""

    if srt_source == "SRT ဖိုင်တင်မယ်":

        if not edit_srt_file:

            st.error(
                "SRT ဖိုင်တင်ပါ။"
            )

            st.stop()

        edit_srt_text = (
            edit_srt_file
            .getvalue()
            .decode(
                "utf-8-sig",
                errors="replace",
            )
        )

    elif srt_source == "Step 1 မှာထုတ်ထားတာသုံးမယ်":

        edit_srt_text = (
            st.session_state.get(
                "srt_text",
                "",
            )
        )

    # --------------------------------------------------------
    # Parse SRT
    # --------------------------------------------------------

    if edit_srt_text.strip():

        try:

            edit_segments = parse_srt(
                edit_srt_text
            )

        except Exception as e:

            st.error(
                "SRT timing မမှန်ပါ။"
            )

            st.exception(e)

            st.stop()

    else:

        edit_segments = []

    # --------------------------------------------------------
    # Prepare working directory
    # --------------------------------------------------------

    try:

        with tempfile.TemporaryDirectory() as td:

            work = Path(td)

            source_video = (
                work
                / "input_video.mp4"
            )

            source_video.write_bytes(
                edit_video_file.getbuffer()
            )

            # ------------------------------------------------
            # Voiceover
            # ------------------------------------------------

            voice_path = None

            if (
                voice_source
                == "Step 2 မှာထုတ်ထားတာသုံးမယ်"
            ):

                voice_bytes = (
                    st.session_state.get(
                        "voice_bytes"
                    )
                )

                if voice_bytes:

                    voice_path = (
                        work
                        / "voiceover.m4a"
                    )

                    voice_path.write_bytes(
                        voice_bytes
                    )

                else:

                    st.error(
                        "Step 2 Voiceover မတွေ့ပါ။"
                    )

                    st.stop()

            elif (
                voice_source
                == "Voiceover ဖိုင်တင်မယ်"
            ):

                if not edit_voice_file:

                    st.error(
                        "Voiceover ဖိုင်တင်ပါ။"
                    )

                    st.stop()

                voice_path = (
                    work
                    / "voiceover_input"
                )

                voice_path.write_bytes(
                    edit_voice_file.getbuffer()
                )

            # ------------------------------------------------
            # Music
            # ------------------------------------------------

            music_path = None

            if music_file:

                music_path = (
                    work
                    / "background_music"
                )

                music_path.write_bytes(
                    music_file.getbuffer()
                )

            # ------------------------------------------------
            # Output
            # ------------------------------------------------

            output_name = str(
                final_filename
                or "Myanmar_Movie_Final"
            ).strip()

            output_name = re.sub(
                r'[\\/:*?"<>|]+',
                "_",
                output_name,
            )

            output_name = re.sub(
                r"\s+",
                "_",
                output_name,
            )

            if output_name.lower().endswith(
                ".mp4"
            ):
                output_name = output_name[:-4]

            if not output_name:
                output_name = (
                    "Myanmar_Movie_Final"
                )

            output_path = (
                work
                / f"{output_name}.mp4"
            )

            status = st.empty()

            bar = st.progress(
                0.0
            )

            status.info(
                "🎬 Video editing စတင်နေသည်..."
            )

            bar.progress(
                0.10
            )

            # ------------------------------------------------
            # Render
            # ------------------------------------------------

            status.info(
                "✂️ Trim / Ratio / Flip / Zoom "
                "ပြုလုပ်နေသည်..."
            )

            bar.progress(
                0.20
            )

            result_video = build_edit_video(
                video_path=source_video,
                output_path=output_path,
                trim_start=trim_start,
                trim_end=trim_end,
                ratio=edit_ratio,
                flip_h=flip_h,
                flip_v=flip_v,
                zoom=zoom,
                pos_x=pos_x,
                pos_y=pos_y,
                blur_background=blur_background,
                mask_type=mask_type,
                srt_segments=edit_segments,
                subtitle_size=subtitle_size,
                subtitle_color=subtitle_color,
                subtitle_position=subtitle_position,
                subtitle_outline=subtitle_outline,
                voice_path=voice_path,
                music_path=music_path,
                original_audio=original_audio,
                original_volume=original_volume,
                voice_volume=voice_volume,
                music_volume=music_volume,
                quality=quality,
            )

            bar.progress(
                0.95
            )

            status.info(
                "🔍 Final MP4 စစ်ဆေးနေသည်..."
            )

            final_bytes = (
                result_video.read_bytes()
            )

            if len(final_bytes) < 10000:

                raise RuntimeError(
                    "Final MP4 ဖိုင်အရွယ်အစား မမှန်ပါ။"
                )

            # ------------------------------------------------
            # Save in session
            # ------------------------------------------------

            st.session_state[
                "final_video_bytes"
            ] = final_bytes

            st.session_state[
                "final_video_name"
            ] = f"{output_name}.mp4"

            bar.progress(
                1.0
            )

            status.success(
                "✅ Final Video ပြီးပါပြီ!"
            )

    except Exception as e:

        st.error(
            "❌ Final Video ထုတ်ရာမှာ "
            "အမှားဖြစ်ပါတယ်။"
        )

        st.exception(e)


# ============================================================
# FINAL VIDEO RESULT
# ============================================================

if st.session_state.get(
    "final_video_bytes"
):

    st.subheader(
        "🎬 Final Video Preview"
    )

    st.video(
        st.session_state[
            "final_video_bytes"
        ]
    )

    st.download_button(
        "⬇️ DOWNLOAD FINAL MP4",
        st.session_state[
            "final_video_bytes"
        ],
        st.session_state.get(
            "final_video_name",
            "Myanmar_Movie_Final.mp4",
        ),
        "video/mp4",
        use_container_width=True,
        key="download_final_video",
    )
