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


# ============================================================
# UI
# ============================================================

st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(91, 76, 255, .14), transparent 32%),
            radial-gradient(circle at 90% 5%, rgba(0, 200, 255, .10), transparent 28%),
            #080b12;
        color: #f4f7fb;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(25,31,48,.96), rgba(12,16,26,.96));
        box-shadow: 0 18px 50px rgba(0,0,0,.28);
        margin-bottom: 22px;
    }

    .hero h1 {
        margin: 0 0 7px 0;
        font-size: clamp(30px, 5vw, 48px);
        letter-spacing: -1.2px;
    }

    .hero p {
        margin: 0;
        color: #aeb8ca;
        font-size: 15px;
    }

    .section-card {
        padding: 7px 0 0 0;
    }

    .mini {
        color: #9aa6bb;
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 16px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 16px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 14px;
        min-height: 48px;
        font-weight: 700;
    }

    .status-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.08);
        color: #cbd4e5;
        font-size: 12px;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .footer {
        text-align: center;
        color: #69758a;
        font-size: 12px;
        padding-top: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
    <h1>🎬 Myanmar Movie AI</h1>
    <p>Movie dialogue ကို မြန်မာ SRT အဖြစ်ပြောင်းပြီး သဘာဝကျ Burmese Voiceover ထုတ်ပေးတဲ့ Studio</p>
    <div style="margin-top:16px">
        <span class="status-pill">🎙️ Deepgram STT</span>
        <span class="status-pill">🤖 Gemini Translation</span>
        <span class="status-pill">🗣️ Burmese Neural Voice</span>
        <span class="status-pill">⚡ Auto Timestamp</span>
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
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    return value or default


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
        [FFMPEG, "-hide_banner", "-i", str(path)],
        timeout=120,
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr or "",
    )

    if not match:
        raise RuntimeError("Media duration ကို ဖတ်မရပါ။")

    return (
        int(match.group(1)) * 3600
        + int(match.group(2)) * 60
        + float(match.group(3))
    )


def extract_audio(video_path: Path, audio_path: Path):
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


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


# ============================================================
# GEMINI TRANSLATION
# ============================================================

def get_gemini_client():
    key = get_secret("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return genai.Client(api_key=key)


def get_gemini_model() -> str:
    return get_secret("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


def extract_json_array(text: str):
    value = (text or "").strip()

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

    return json.loads(value[start:end + 1])


def translate_batch(client, rows):
    payload = [
        {
            "id": i + 1,
            "text": row["source"],
        }
        for i, row in enumerate(rows)
    ]

    prompt = f"""
You are a professional Myanmar movie dubbing subtitle translator.

Translate each dialogue into natural, conversational Burmese used by real Myanmar speakers.

Rules:
- Preserve the complete original meaning and intent.
- Preserve names and important proper nouns.
- Preserve emotion, tone, attitude and context.
- Do not summarize.
- Do not remove meaningful information.
- Do not add explanations.
- Do not add quotation marks unless they are part of the meaning.
- Make the Burmese sound natural when spoken aloud by a Myanmar voice actor.
- Keep the sentence coherent and connected to the original meaning.
- Do not truncate the Burmese just to make it shorter.
- Return ONLY a JSON array.
- Return exactly {len(payload)} objects.
- Keep the exact id values.

Format:
[{{"id":1,"burmese":"..."}}]

INPUT:
{json.dumps(payload, ensure_ascii=False)}
"""

    model = get_gemini_model()
    last_error = ""

    # 503 အတွက် 8 → 16 → 32 → 60 → 60 seconds
    # စုစုပေါင်း 5 attempts
    retry_delays_503 = [8, 16, 32, 60]

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            data = extract_json_array(
                getattr(response, "text", "")
            )

            if (
                not isinstance(data, list)
                or len(data) != len(payload)
            ):
                raise RuntimeError(
                    "Gemini translation result count မကိုက်ပါ။"
                )

            translated = {}

            for item in data:
                idx = int(item["id"])
                text = clean_text(
                    item.get("burmese", "")
                )

                if text:
                    translated[idx] = text

            if len(translated) != len(payload):
                raise RuntimeError(
                    "ဘာသာပြန်စာကြောင်းတချို့ မထွက်ပါ။"
                )

            return translated

        except Exception as exc:
            last_error = str(exc)
            low = last_error.lower()

            is_503 = (
                "503" in low
                or "unavailable" in low
                or "high demand" in low
                or "overloaded" in low
            )

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
                ]
            )

            if transient and attempt < 4:

                if is_503:
                    delay = retry_delays_503[attempt]
                else:
                    delay = [2, 4, 8, 16][attempt]

                time.sleep(
                    delay + random.random()
                )

            else:
                break

    raise RuntimeError(
        f"Gemini translation မအောင်မြင်ပါ။\n{last_error}"
    )


# ============================================================
# DEEPGRAM TRANSCRIPTION + SEGMENTATION
# ============================================================

def get_deepgram_key():
    key = get_secret("DEEPGRAM_API_KEY")

    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY မတွေ့ပါ။ Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return key


def words_to_segments(words):
    max_chars = 42
    max_duration = 6.0
    pause_split = 0.65

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
            item.get("punctuated_word")
            or item.get("word")
            or ""
        ).strip()

    def flush():
        nonlocal current

        if not current:
            return

        first = current[0]
        last = current[-1]

        start = float(
            first.get("start", 0.0)
        )

        end = float(
            last.get("end", start)
        )

        text = clean_text(
            " ".join(
                word_text(x)
                for x in current
            )
        )

        if text and end > start:
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
                    for x in current + [word]
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
                or len(proposed) > max_chars
                or duration > max_duration
            ):
                flush()

        current.append(word)

        if word_text(word).endswith(
            punctuation
        ):
            flush()

    flush()

    return segments


def deepgram_transcribe(audio_path: Path):
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
        "Authorization": f"Token {get_deepgram_key()}",
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
                    segments = words_to_segments(
                        words
                    )

                    if segments:
                        return segments

                utterances = results.get(
                    "utterances",
                    [],
                ) or []

                fallback = []

                for item in utterances:

                    start = float(
                        item.get(
                            "start",
                            0.0,
                        )
                    )

                    end = float(
                        item.get(
                            "end",
                            start,
                        )
                    )

                    text = clean_text(
                        item.get(
                            "transcript",
                            "",
                        )
                    )

                    if text and end > start:
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
                    "Deepgram transcript မထွက်ပါ။"
                )

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )

            if (
                response.status_code in RETRY_STATUS
                and attempt < 2
            ):
                time.sleep(
                    2 ** attempt
                )
            else:
                break

        except Exception as exc:

            last_error = str(exc)

            if attempt < 2:
                time.sleep(
                    2 ** attempt
                )
            else:
                break

    raise RuntimeError(
        "Deepgram transcription မအောင်မြင်ပါ။\n"
        + last_error
    )


# ============================================================
# SRT BUILDING
# ============================================================

def build_srt_segments(
    client,
    source_segments,
    progress_callback,
):
    batch_size = 12

    translated_all = []

    total = len(source_segments)

    if total == 0:
        raise RuntimeError(
            "Dialogue မတွေ့ပါ။"
        )

    for start_index in range(
        0,
        total,
        batch_size,
    ):

        batch = source_segments[
            start_index:start_index + batch_size
        ]

        translated = translate_batch(
            client,
            batch,
        )

        for index, row in enumerate(
            batch
        ):

            text = translated.get(
                index + 1,
                "",
            )

            if not text:
                raise RuntimeError(
                    "ဘာသာပြန်စာကြောင်းတစ်ခု မထွက်ပါ။"
                )

            translated_all.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "source": row["source"],
                    "burmese": text,
                }
            )

        progress_callback(
            min(
                0.25
                + 0.70
                * (
                    len(translated_all)
                    / total
                ),
                0.95,
            ),
            (
                f"ဘာသာပြန်နေသည်... "
                f"{len(translated_all)}/{total}"
            ),
        )

    return translated_all


def srt_timestamp(seconds: float):
    seconds = max(
        0.0,
        float(seconds),
    )

    total_ms = int(
        round(seconds * 1000)
    )

    hours = total_ms // 3600000
    total_ms %= 3600000

    minutes = total_ms // 60000
    total_ms %= 60000

    secs = total_ms // 1000
    millis = total_ms % 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{millis:03d}"
    )


def make_srt(segments):
    blocks = []

    for index, item in enumerate(
        segments,
        start=1,
    ):

        blocks.append(
            "\n".join(
                [
                    str(index),
                    (
                        f"{srt_timestamp(item['start'])}"
                        f" --> "
                        f"{srt_timestamp(item['end'])}"
                    ),
                    item["burmese"],
                ]
            )
        )

    return "\n\n".join(blocks)


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

    segments = []
    fixed_count = 0

    timestamp_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3})"
        r"\s*-->\s*"
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3})"
    )

    def parse_time(value):
        value = value.replace(
            ",",
            ".",
        )

        parts = value.split(":")

        if len(parts) != 3:
            return 0.0

        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])

        return (
            hours * 3600
            + minutes * 60
            + seconds
        )

    for block in blocks:

        lines = [
            line.strip()
            for line in block.split("\n")
            if line.strip()
        ]

        if len(lines) < 2:
            continue

        match = None
        time_line_index = -1

        for i, line in enumerate(lines):

            candidate = timestamp_pattern.search(
                line
            )

            if candidate:
                match = candidate
                time_line_index = i
                break

        if not match:
            continue

        start = parse_time(
            match.group(1)
        )

        end = parse_time(
            match.group(2)
        )

        text_lines = lines[
            time_line_index + 1:
        ]

        burmese = clean_text(
            " ".join(text_lines)
        )

        if not burmese:
            continue

        if end <= start:
            end = start + 0.5
            fixed_count += 1

        segments.append(
            {
                "start": start,
                "end": end,
                "burmese": burmese,
            }
        )

    segments.sort(
        key=lambda x: x["start"]
    )

    for i in range(
        1,
        len(segments),
    ):

        previous = segments[i - 1]
        current = segments[i]

        if current["start"] < previous["start"]:
            current["start"] = previous["start"]
            fixed_count += 1

        if current["end"] <= current["start"]:
            current["end"] = (
                current["start"] + 0.5
            )
            fixed_count += 1

    if not segments:
        raise RuntimeError(
            "SRT ထဲမှာ အသုံးပြုလို့ရတဲ့ subtitle မတွေ့ပါ။"
        )

    return segments, fixed_count


# ============================================================
# TTS
# ============================================================

def make_tts(
    text,
    voice,
    style,
    output,
):
    style_config = VOICE_STYLES.get(
        style,
        VOICE_STYLES["ပုံမှန်"],
    )

    rate = style_config["rate"]
    pitch = style_config["pitch"]

    rate_string = (
        f"{rate:+d}%"
        if rate
        else "+0%"
    )

    pitch_string = (
        f"{pitch:+d}Hz"
        if pitch
        else "+0Hz"
    )

    async def generate():
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate_string,
            pitch=pitch_string,
        )

        await communicate.save(
            str(output)
        )

    try:
        asyncio.run(
            generate()
        )

    except RuntimeError:
        loop = asyncio.new_event_loop()

        try:
            loop.run_until_complete(
                generate()
            )
        finally:
            loop.close()

    if (
        not output.exists()
        or output.stat().st_size < 1000
    ):
        raise RuntimeError(
            "Burmese TTS audio မထွက်ပါ။"
        )


# ============================================================
# AUDIO TIMING
# ============================================================

def atempo_chain(factor):
    factor = max(
        0.25,
        min(
            float(factor),
            4.0,
        ),
    )

    parts = []

    while factor > 2.0:
        parts.append(
            "atempo=2.0"
        )
        factor /= 2.0

    while factor < 0.5:
        parts.append(
            "atempo=0.5"
        )
        factor /= 0.5

    parts.append(
        f"atempo={factor:.6f}"
    )

    return ",".join(parts)


def fit_tts_to_slot(
    source,
    output,
    slot,
    user_speed,
):
    slot = max(
        0.08,
        float(slot),
    )

    probe = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(source),
        ],
        timeout=120,
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        probe.stderr or "",
    )

    if not match:
        raise RuntimeError(
            "TTS audio duration ကို ဖတ်မရပါ။"
        )

    raw_duration = (
        int(match.group(1)) * 3600
        + int(match.group(2)) * 60
        + float(match.group(3))
    )

    if raw_duration <= 0:
        raise RuntimeError(
            "TTS audio duration မမှန်ပါ။"
        )

    factor = (
        max(
            1.0,
            raw_duration / slot,
        )
        * user_speed
    )

    factor = max(
        0.25,
        min(
            factor,
            4.0,
        ),
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
            (
                atempo_chain(factor)
                + ",apad,"
                + f"atrim=duration={slot:.3f}"
            ),
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
            + (result.stderr or "")
        )


# ============================================================
# BUILD VOICEOVER
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

    if total == 0:
        raise RuntimeError(
            "Voiceover ပြုလုပ်ရန် SRT line မရှိပါ။"
        )

    for index, item in enumerate(
        segments,
        start=1,
    ):

        start = max(
            0.0,
            float(item["start"]),
        )

        end = max(
            start + 0.08,
            float(item["end"]),
        )

        slot = end - start

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
            f"Voice {index}/{total} ထုတ်နေသည်...",
        )

        make_tts(
            item["burmese"],
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

    for index, (start, _) in enumerate(
        clips
    ):

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

    # ========================================================
    # IMPORTANT:
    # Final audio MUST NOT continue beyond the last SRT time.
    # This fixes the 12-minute output problem.
    # ========================================================

    timeline_end = max(
        float(item["end"])
        for item in segments
    )

    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:"
        + "duration=longest:"
        + "dropout_transition=0,"
        + f"atrim=duration={timeline_end:.3f},"
        + "asetpts=PTS-STARTPTS,"
        + "loudnorm=I=-16:TP=-1.5:LRA=11,"
        + "volume=1.25,"
        + "alimiter=limit=0.95"
        + "[out]"
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
            + (result.stderr or "")
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
            "Voiceover audio validation မအောင်မြင်ပါ။\n"
            + (validation.stderr or "")
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
    st.session_state.srt_name = "myanmar.srt"

if "voice_bytes" not in st.session_state:
    st.session_state.voice_bytes = None

if "voice_name" not in st.session_state:
    st.session_state.voice_name = (
        "myanmar_voiceover.m4a"
    )


# ============================================================
# STEP 1 — VIDEO -> SRT
# ============================================================

st.markdown(
    "## ① Video → မြန်မာ SRT"
)

st.markdown(
    '<div class="mini">'
    "Audio ကို အလိုအလျောက်ထုတ် → dialogue timestamp ခွဲ "
    "→ Gemini နဲ့ သဘာဝကျ မြန်မာလိုဘာသာပြန် → SRT"
    "</div>",
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
            Path(video_file.name).stem
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
                "🎧 Video audio ထုတ်နေသည်..."
            )

            extract_audio(
                video_path,
                audio_path,
            )

            progress.progress(
                0.12
            )

            status.info(
                "🎙️ Deepgram က dialogue + word timestamp ရယူနေသည်..."
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
                "🤖 Gemini က မြန်မာလို သဘာဝကျ ဘာသာပြန်နေသည်..."
            )

            translated_segments = (
                build_srt_segments(
                    get_gemini_client(),
                    source_segments,
                    lambda p, text: (
                        progress.progress(
                            min(
                                p,
                                0.98,
                            )
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

            if not st.session_state.srt_name.lower().endswith(
                ".srt"
            ):
                st.session_state.srt_name += ".srt"

            progress.progress(
                1.0
            )

            status.success(
                "✅ SRT ပြီးပါပြီ — "
                f"{len(translated_segments)} subtitle lines"
            )

    except Exception as exc:

        st.error(
            "SRT ထုတ်ရာမှာ အမှားဖြစ်ပါတယ်။"
        )

        st.exception(exc)


if st.session_state.srt_text:

    st.markdown(
        "### 📄 Myanmar SRT Preview"
    )

    st.text_area(
        "",
        st.session_state.srt_text,
        height=280,
        label_visibility="collapsed",
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
# STEP 2 — SRT -> VOICEOVER
# ============================================================

st.markdown("---")

st.markdown(
    "## ② SRT → မြန်မာ Voiceover"
)

st.markdown(
    '<div class="mini">'
    "SRT timestamp ကို အလိုအလျောက်စစ်/ပြင်ပြီး "
    "subtitle timing အတိုင်း Burmese voiceover တည်ဆောက်ပေးပါတယ်။"
    "</div>",
    unsafe_allow_html=True,
)

srt_file = st.file_uploader(
    "📄 SRT ဖိုင်တင်ပါ (သို့) အဆင့် ၁ က SRT ကို တိုက်ရိုက်သုံးပါ",
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
            list(
                VOICES.keys()
            ),
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
            value="myanmar_voiceover.m4a",
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
            "SRT ဖိုင်တင်ပါ (သို့) အဆင့် ၁ မှာ SRT အရင်ထုတ်ပါ။"
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
                "⏱️ SRT timing ကို အလိုအလျောက်ပြင်ပြီးပါပြီ — "
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

            voice_path = (
                build_voiceover(
                    segments,
                    VOICES[
                        selected_voice
                    ],
                    selected_style,
                    selected_speed,
                    work,
                    lambda p, text: (
                        progress.progress(
                            min(
                                p,
                                1.0,
                            )
                        ),
                        status.info(
                            text
                        ),
                    ),
                )
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
                "✅ Voiceover ပြီးပါပြီ — "
                "SRT timing အတိုင်း audio ပြုလုပ်ပြီးပါပြီ။"
            )

    except Exception as exc:

        st.error(
            "Voiceover ထုတ်ရာမှာ အမှားဖြစ်ပါတယ်။"
        )

        st.exception(exc)


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
        file_name=st.session_state.voice_name,
        mime="audio/mp4",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    "🎬 Myanmar Movie AI · "
    "Video → Myanmar SRT → Burmese Voiceover"
    "</div>",
    unsafe_allow_html=True,
)
