import os
import re
import json
import time
import random
import base64
import subprocess
import tempfile
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
import edge_tts
import asyncio
import imageio_ffmpeg
from google import genai


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Myanmar Movie AI Studio",
    page_icon="🎬",
    layout="wide",
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

EDITOR_DIR = Path(__file__).parent / "video_editor"


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
# BASIC HELPERS
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


def probe_video_size(path):
    r = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(path),
        ],
        120,
    )

    text = r.stderr or ""

    patterns = [
        r"Video:.*?(\d{2,5})x(\d{2,5})",
        r"Stream.*Video.*?(\d{2,5})x(\d{2,5})",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.I | re.S)
        if m:
            return int(m.group(1)), int(m.group(2))

    raise RuntimeError("Video resolution ကို ဖတ်မရပါ။")


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


def get_secret(name):
    value = st.secrets.get(name, "")

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

    return name or "output"


# ============================================================
# GEMINI
# ============================================================

def gemini_client():
    key = get_secret("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return genai.Client(api_key=key)


def clean_text(s):
    return re.sub(
        r"\s+",
        " ",
        str(s or ""),
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

    a = text.find("[")
    b = text.rfind("]")

    if a >= 0 and b > a:
        return text[a:b + 1]

    return text


def translate_batch(client, items):
    prompt = (
        """
You are a professional movie subtitle translator.

Translate every source dialogue into natural conversational Burmese.

Rules:
- Preserve meaning.
- Preserve names.
- Preserve emotion.
- Do not add explanations.
- Do not summarize.
- Keep it concise enough for the original subtitle timing.
- Return ONLY JSON.
- Return exactly the same number of objects.
- Keep the same id values.

Format:
[
  {"id":1,"burmese":"..."}
]

INPUT:
"""
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

                data = json.loads(
                    clean_json(
                        getattr(
                            response,
                            "text",
                            "",
                        )
                    )
                )

                if (
                    not isinstance(data, list)
                    or len(data) != len(items)
                ):
                    raise RuntimeError(
                        "Gemini translation result count မကိုက်ပါ။"
                    )

                result = {}

                for item in data:
                    idx = int(item["id"])
                    text = clean_text(
                        item.get("burmese", "")
                    )

                    if text:
                        result[idx] = text

                for i in range(
                    1,
                    len(items) + 1,
                ):
                    if i not in result:
                        raise RuntimeError(
                            "ဘာသာပြန်စာကြောင်းတချို့ မထွက်ပါ။"
                        )

                return result

            except Exception as e:
                errors.append(
                    f"{model} attempt {attempt + 1}: {e}"
                )

                if (
                    attempt == 0
                    and any(
                        x in str(e).lower()
                        for x in RETRY_WORDS
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


# ============================================================
# DEEPGRAM
# ============================================================

def deepgram_key():
    key = get_secret("DEEPGRAM_API_KEY")

    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return key


def words_to_segments(words):
    """
    Deepgram word timestamps -> short subtitle segments.

    Prevents one giant paragraph from becoming one subtitle.
    """

    MAX_CHARS = 42
    MAX_DURATION = 6.0
    PAUSE_SPLIT = 0.65

    result = []
    current = []

    def flush():
        nonlocal current

        if not current:
            return

        first = current[0]
        last = current[-1]

        text = " ".join(
            str(
                x.get(
                    "punctuated_word",
                    x.get("word", ""),
                )
            )
            for x in current
        ).strip()

        start = float(
            first.get("start", 0)
        )

        end = float(
            last.get(
                "end",
                last.get("start", start),
            )
        )

        if text and end > start:
            result.append(
                {
                    "start": start,
                    "end": end,
                    "source": clean_text(text),
                }
            )

        current = []

    for word in words:
        if not word.get("word"):
            continue

        if current:
            prev_end = float(
                current[-1].get(
                    "end",
                    current[-1].get("start", 0),
                )
            )

            cur_start = float(
                word.get("start", prev_end)
            )

            pause = cur_start - prev_end

            existing_text = " ".join(
                str(
                    x.get(
                        "punctuated_word",
                        x.get("word", ""),
                    )
                )
                for x in current
            )

            proposed_text = (
                existing_text
                + " "
                + str(
                    word.get(
                        "punctuated_word",
                        word.get("word", ""),
                    )
                )
            )

            proposed_duration = (
                float(
                    word.get(
                        "end",
                        cur_start,
                    )
                )
                - float(
                    current[0].get(
                        "start",
                        cur_start,
                    )
                )
            )

            if (
                pause >= PAUSE_SPLIT
                or len(proposed_text) > MAX_CHARS
                or proposed_duration > MAX_DURATION
            ):
                flush()

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
            flush()

    flush()

    return result


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
            response = requests.post(
                url,
                params=params,
                headers=headers,
                data=data,
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
                )

                words = []

                if channels:
                    alternatives = channels[0].get(
                        "alternatives",
                        [],
                    )

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

                for u in utterances:
                    text = clean_text(
                        u.get(
                            "transcript",
                            "",
                        )
                    )

                    start = float(
                        u.get("start", 0)
                    )

                    end = float(
                        u.get("end", start)
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
                    "Deepgram က transcript မပြန်ပေးပါ။"
                )

            last = (
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
# BUILD SRT
# ============================================================

def build_segments(
    client,
    source_segments,
    progress,
):
    rows = []

    for item in source_segments:
        text = clean_text(
            item.get("source", "")
        )

        start = float(
            item.get("start", 0)
        )

        end = float(
            item.get("end", start)
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

    result = []

    batch_size = 12

    for pos in range(
        0,
        len(rows),
        batch_size,
    ):
        batch = rows[
            pos:pos + batch_size
        ]

        payload = [
            {
                "id": i + 1,
                "text": item["source"],
            }
            for i, item in enumerate(batch)
        ]

        progress(
            0.2
            + 0.45
            * (
                pos
                / max(len(rows), 1)
            ),
            (
                "Gemini ဘာသာပြန်နေသည်... "
                f"{min(pos + len(batch), len(rows))}"
                f"/{len(rows)}"
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
            result.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "burmese": translated[i],
                }
            )

    return result


def srt_time(seconds):
    seconds = max(
        0.0,
        float(seconds),
    )

    milliseconds = int(
        round(seconds * 1000)
    )

    hours, rem = divmod(
        milliseconds,
        3600000,
    )

    minutes, rem = divmod(
        rem,
        60000,
    )

    secs, milliseconds = divmod(
        rem,
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

    for i, item in enumerate(
        segments,
        1,
    ):
        blocks.append(
            f"{i}\n"
            f"{srt_time(item['start'])} --> "
            f"{srt_time(item['end'])}\n"
            f"{item['burmese']}\n"
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
            h, m, s = (
                value
                .replace(",", ".")
                .split(":")
            )

            return (
                int(h) * 3600
                + int(m) * 60
                + float(s)
            )

        idx = lines.index(time_line)

        subtitle_lines = [
            x.strip()
            for x in lines[idx + 1:]
            if x.strip()
        ]

        subtitle = clean_text(
            " ".join(subtitle_lines)
        )

        if subtitle:
            output.append(
                {
                    "start": parse_time(
                        match.group(1)
                    ),
                    "end": parse_time(
                        match.group(2)
                    ),
                    "burmese": subtitle,
                }
            )

    output.sort(
        key=lambda x: x["start"]
    )

    cleaned = []

    for item in output:
        if (
            item["end"]
            <= item["start"]
        ):
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
            "SRT ထဲမှာ valid subtitle မတွေ့ပါ။"
        )

    return cleaned


# ============================================================
# TTS
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
    config = VOICE_STYLES[style]

    errors = []

    for attempt in range(3):
        try:
            if out.exists():
                out.unlink()

            asyncio.run(
                edge_tts_save(
                    text,
                    voice,
                    config["rate"],
                    config["pitch"],
                    out,
                )
            )

            if (
                out.exists()
                and out.stat().st_size > 1000
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
            (
                atempo_chain(final_speed)
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

    for i, item in enumerate(
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

        raw = (
            work
            / f"tts_{i:04d}.mp3"
        )

        fitted = (
            work
            / f"clip_{i:04d}.m4a"
        )

        progress(
            0.1
            + 0.65
            * ((i - 1) / total),
            f"Voice {i}/{total} ထုတ်နေသည်...",
        )

        make_tts(
            item["burmese"],
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

    for _, file in clips:
        command += [
            "-i",
            str(file),
        ]

    filters = []
    labels = []

    for i, (start, _) in enumerate(
        clips
    ):
        delay = max(
            0,
            int(round(start * 1000)),
        )

        label = f"a{i}"

        filters.append(
            f"[{i}:a]"
            f"adelay={delay}:all=1,"
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
          "loudnorm=I=-16:TP=-1.5:LRA=11,"
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

    r = run_cmd(
        command,
        1800,
    )

    if (
        r.returncode
        or not output.exists()
        or output.stat().st_size < 5000
    ):
        raise RuntimeError(
            "Voiceover file မထုတ်နိုင်ပါ။\n"
            + (r.stderr or "")
        )

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

    if check.returncode:
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


# ============================================================
# VIDEO EDITOR PREVIEW
# ============================================================

def make_preview_video(
    source,
    output,
):
    """
    Small silent proxy video for the browser editor.
    """

    r = run_cmd(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            (
                "scale=720:720:"
                "force_original_aspect_ratio=decrease,"
                "pad=720:720:"
                "(ow-iw)/2:"
                "(oh-ih)/2"
            ),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-movflags",
            "+faststart",
            str(output),
        ],
        900,
    )

    if (
        r.returncode
        or not output.exists()
    ):
        raise RuntimeError(
            "Editor preview video မထုတ်နိုင်ပါ။\n"
            + (r.stderr or "")
        )


def video_data_url(path):
    data = path.read_bytes()

    encoded = base64.b64encode(
        data
    ).decode("ascii")

    return (
        "data:video/mp4;base64,"
        + encoded
    )


# ============================================================
# ASS SUBTITLE
# ============================================================

def ass_escape(text):
    text = str(text or "")

    text = text.replace(
        "\\",
        r"\\",
    )

    text = text.replace(
        "{",
        r"\{",
    )

    text = text.replace(
        "}",
        r"\}",
    )

    text = text.replace(
        "\n",
        r"\N",
    )

    return text


def ass_time(seconds):
    seconds = max(
        0.0,
        float(seconds),
    )

    h = int(seconds // 3600)

    seconds -= h * 3600

    m = int(seconds // 60)

    seconds -= m * 60

    s = int(seconds)

    cs = int(
        round(
            (seconds - s) * 100
        )
    )

    if cs >= 100:
        cs = 0
        s += 1

    return (
        f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    )


def make_ass(
    segments,
    width,
    height,
    state,
):
    subtitle = state.get(
        "subtitle",
        {},
    )

    x = float(
        subtitle.get(
            "x",
            0.5,
        )
    )

    y = float(
        subtitle.get(
            "y",
            0.86,
        )
    )

    size = int(
        subtitle.get(
            "size",
            42,
        )
    )

    outline = int(
        subtitle.get(
            "outline",
            3,
        )
    )

    pos_x = int(
        x * width
    )

    pos_y = int(
        y * height
    )

    font_size = max(
        18,
        min(size, 120),
    )

    outline = max(
        0,
        min(outline, 12),
    )

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, "
        "PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            "Style: Default,Noto Sans Myanmar,"
            f"{font_size},"
            "&H00FFFFFF,"
            "&H000000FF,"
            "&H00000000,"
            "&H80000000,"
            "0,0,0,0,100,100,0,0,"
            f"1,{outline},0,5,20,20,20,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, "
        "Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]

    for item in segments:
        text = ass_escape(
            item["burmese"]
        )

        dialogue = (
            "Dialogue: 0,"
            f"{ass_time(item['start'])},"
            f"{ass_time(item['end'])},"
            "Default,,0,0,0,"
            f"{{\\pos({pos_x},{pos_y})}}"
            f"{text}"
        )

        lines.append(dialogue)

    return "\n".join(lines)


# ============================================================
# VIDEO CANVAS
# ============================================================

def canvas_size(ratio, iw, ih):
    ratio = str(ratio or "Original")

    if ratio == "9:16":
        return 1080, 1920

    if ratio == "16:9":
        return 1920, 1080

    if ratio == "1:1":
        return 1080, 1080

    if ratio == "4:5":
        return 1080, 1350

    return iw, ih


def build_canvas_filter(
    iw,
    ih,
    cw,
    ch,
    state,
):
    zoom = float(
        state.get(
            "zoom",
            1.0,
        )
    )

    zoom = max(
        0.5,
        min(zoom, 3.0),
    )

    flip_h = bool(
        state.get(
            "flip_h",
            False,
        )
    )

    flip_v = bool(
        state.get(
            "flip_v",
            False,
        )
    )

    if flip_h and flip_v:
        flip_filter = "hflip,vflip"
    elif flip_h:
        flip_filter = "hflip"
    elif flip_v:
        flip_filter = "vflip"
    else:
        flip_filter = ""

    if cw == iw and ch == ih:
        base_scale = zoom
    else:
        base_scale = max(
            cw / iw,
            ch / ih,
        ) * zoom

    sw = max(
        2,
        int(round(iw * base_scale)),
    )

    sh = max(
        2,
        int(round(ih * base_scale)),
    )

    pos = state.get(
        "position",
        {},
    )

    px = float(
        pos.get(
            "x",
            0.0,
        )
    )

    py = float(
        pos.get(
            "y",
            0.0,
        )
    )

    x = int(
        (cw - sw) / 2
        + px * cw
    )

    y = int(
        (ch - sh) / 2
        + py * ch
    )

    chain = []

    if flip_filter:
        chain.append(
            flip_filter
        )

    chain.append(
        f"scale={sw}:{sh}"
    )

    chain.append(
        "format=yuv420p"
    )

    chain.append(
        f"pad={cw}:{ch}:{x}:{y}:black"
    )

    return ",".join(chain)


# ============================================================
# MASK FILTER
# ============================================================

def apply_mask_filter(
    label,
    cw,
    ch,
    state,
):
    mask = state.get(
        "mask",
        {},
    )

    enabled = bool(
        mask.get(
            "enabled",
            False,
        )
    )

    if not enabled:
        return (
            f"[{label}]"
            "[vmasked]"
        ), []


    mx = max(
        0,
        min(
            int(
                float(
                    mask.get(
                        "x",
                        0.70,
                    )
                ) * cw
            ),
            cw - 1,
        ),
    )

    my = max(
        0,
        min(
            int(
                float(
                    mask.get(
                        "y",
                        0.78,
                    )
                ) * ch
            ),
            ch - 1,
        ),
    )

    mw = max(
        2,
        min(
            int(
                float(
                    mask.get(
                        "w",
                        0.25,
                    )
                ) * cw
            ),
            cw - mx,
        ),
    )

    mh = max(
        2,
        min(
            int(
                float(
                    mask.get(
                        "h",
                        0.12,
                    )
                ) * ch
            ),
            ch - my,
        ),
    )

    mode = str(
        mask.get(
            "type",
            "solid",
        )
    ).lower()

    invert = bool(
        mask.get(
            "invert",
            False,
        )
    )

    filters = []

    if not invert:
        if mode == "blur":
            filters.append(
                f"[{label}]split=2[main][blur0]"
            )

            filters.append(
                "[blur0]"
                f"crop={mw}:{mh}:{mx}:{my},"
                "boxblur=18:2[blur1]"
            )

            filters.append(
                "[main][blur1]"
                f"overlay={mx}:{my}"
                "[vmasked]"
            )

        elif mode == "mosaic":
            filters.append(
                f"[{label}]split=2[main][mos0]"
            )

            small_w = max(
                2,
                mw // 12,
            )

            small_h = max(
                2,
                mh // 12,
            )

            filters.append(
                "[mos0]"
                f"crop={mw}:{mh}:{mx}:{my},"
                f"scale={small_w}:{small_h}:"
                "flags=bilinear,"
                f"scale={mw}:{mh}:flags=neighbor"
                "[mos1]"
            )

            filters.append(
                "[main][mos1]"
                f"overlay={mx}:{my}"
                "[vmasked]"
            )

        else:
            filters.append(
                f"[{label}]"
                f"drawbox=x={mx}:y={my}:"
                f"w={mw}:h={mh}:"
                "color=black@0.92:t=fill"
                "[vmasked]"
            )

    else:
        if mode == "blur":
            filters.append(
                f"[{label}]split=2[main][blur0]"
            )

            filters.append(
                "[blur0]"
                "boxblur=18:2"
                "[blurall]"
            )

            filters.append(
                "[blurall][main]"
                f"crop={mw}:{mh}:{mx}:{my}"
                "[keep]"
            )

            filters.append(
                "[blurall][keep]"
                f"overlay={mx}:{my}"
                "[vmasked]"
            )

        elif mode == "mosaic":
            filters.append(
                f"[{label}]split=2[main][mos0]"
            )

            filters.append(
                "[mos0]"
                "scale=iw/12:ih/12,"
                "scale=iw*12:ih*12:flags=neighbor"
                "[mosall]"
            )

            filters.append(
                "[mosall][main]"
                f"crop={mw}:{mh}:{mx}:{my}"
                "[keep]"
            )

            filters.append(
                "[mosall][keep]"
                f"overlay={mx}:{my}"
                "[vmasked]"
            )

        else:
            filters.append(
                f"[{label}]"
                "drawbox=x=0:y=0:"
                "w=iw:h=ih:"
                "color=black@0.92:t=fill"
                "[maskedall]"
            )

            filters.append(
                "[maskedall]["
                + label
                + "]"
                f"crop={mw}:{mh}:{mx}:{my}"
                "[keep]"
            )

            filters.append(
                "[maskedall][keep]"
                f"overlay={mx}:{my}"
                "[vmasked]"
            )

    return (
        "[vmasked]"
    ), filters


# ============================================================
# FINAL VIDEO RENDER
# ============================================================

def render_manual(
    video_path,
    output_path,
    state,
    srt_segments,
    voice_path=None,
    music_path=None,
    original_volume=0.0,
    voice_volume=1.0,
    music_volume=0.15,
):
    iw, ih = probe_video_size(
        video_path
    )

    ratio = state.get(
        "ratio",
        "Original",
    )

    cw, ch = canvas_size(
        ratio,
        iw,
        ih,
    )

    start = max(
        0.0,
        float(
            state.get(
                "trim_start",
                0.0,
            )
        ),
    )

    duration = ffprobe_duration(
        video_path
    )

    end = float(
        state.get(
            "trim_end",
            duration,
        )
    )

    end = max(
        start + 0.05,
        min(end, duration),
    )

    subtitle_enabled = bool(
        state.get(
            "subtitle_enabled",
            True,
        )
    )

    background_blur = bool(
        state.get(
            "background_blur",
            False,
        )
    )

    canvas_filter = build_canvas_filter(
        iw,
        ih,
        cw,
        ch,
        state,
    )

    filters = []

    filters.append(
        f"[0:v]"
        f"{canvas_filter}"
        "[canvas]"
    )

    video_label = "canvas"

    if background_blur:
        filters.append(
            "[canvas]split=2[sharp][bg0]"
        )

        filters.append(
            "[bg0]"
            "boxblur=18:2"
            "[bg]"
        )

        filters.append(
            "[bg][sharp]"
            "overlay=0:0"
            "[blurcanvas]"
        )

        video_label = "blurcanvas"

    masked_label, mask_filters = (
        apply_mask_filter(
            video_label,
            cw,
            ch,
            state,
        )
    )

    filters.extend(
        mask_filters
    )

    if mask_filters:
        video_label = "vmasked"

    if subtitle_enabled and srt_segments:
        ass_path = (
            Path(tempfile.gettempdir())
            / f"sub_{random.randint(100000,999999)}.ass"
        )

        ass_path.write_text(
            make_ass(
                srt_segments,
                cw,
                ch,
                state,
            ),
            encoding="utf-8",
        )

        filters.append(
            f"[{video_label}]"
            f"subtitles={str(ass_path)}"
            "[vfinal]"
        )

        video_label = "vfinal"

    else:
        filters.append(
            f"[{video_label}]"
            "[vfinal]"
        )

    command = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video_path),
    ]

    audio_inputs = []

    if voice_path and Path(voice_path).exists():
        command += [
            "-i",
            str(voice_path),
        ]

        audio_inputs.append(
            "voice"
        )

    if music_path and Path(music_path).exists():
        command += [
            "-i",
            str(music_path),
        ]

        audio_inputs.append(
            "music"
        )

    command += [
        "-filter_complex",
        ";".join(filters),
    ]

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    audio_filters = []

    audio_maps = []

    input_index = 1

    if voice_path and Path(voice_path).exists():
        audio_filters.append(
            f"[{input_index}:a]"
            f"volume={voice_volume:.3f},"
            "aresample=48000"
            "[voice]"
        )

        audio_maps.append(
            "[voice]"
        )

        input_index += 1

    if music_path and Path(music_path).exists():
        audio_filters.append(
            f"[{input_index}:a]"
            f"volume={music_volume:.3f},"
            "aresample=48000,"
            f"atrim=duration={end-start:.3f}"
            "[music]"
        )

        audio_maps.append(
            "[music]"
        )

    if original_volume > 0.001:
        audio_filters.append(
            "[0:a]"
            f"volume={original_volume:.3f},"
            "aresample=48000"
            "[original]"
        )

        audio_maps.append(
            "[original]"
        )

    if audio_maps:
        if len(audio_maps) == 1:
            audio_filters.append(
                audio_maps[0]
                + "alimiter=limit=0.95"
                "[audiofinal]"
            )
        else:
            audio_filters.append(
                "".join(audio_maps)
                + f"amix=inputs={len(audio_maps)}:"
                  "duration=longest:"
                  "dropout_transition=0,"
                  "loudnorm=I=-16:TP=-1.5:LRA=11,"
                  "alimiter=limit=0.95"
                  "[audiofinal]"
            )

        full_filter = (
            ";".join(filters)
            + ";"
            + ";".join(audio_filters)
        )

        command[
            command.index(
                "-filter_complex"
            ) + 1
        ] = full_filter

        command += [
            "-map",
            "[vfinal]",
            "-map",
            "[audiofinal]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]

    else:
        command += [
            "-map",
            "[vfinal]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

    r = run_cmd(
        command,
        3600,
    )

    if (
        r.returncode
        or not output_path.exists()
        or output_path.stat().st_size < 10000
    ):
        raise RuntimeError(
            "Final MP4 render မအောင်မြင်ပါ။\n"
            + (r.stderr or "")
        )

    return output_path


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_EDITOR_STATE = {
    "ratio": "Original",
    "zoom": 1.0,
    "position": {
        "x": 0.0,
        "y": 0.0,
    },
    "flip_h": False,
    "flip_v": False,
    "background_blur": False,
    "mask": {
        "enabled": False,
        "type": "solid",
        "invert": False,
        "x": 0.68,
        "y": 0.78,
        "w": 0.30,
        "h": 0.13,
    },
    "subtitle_enabled": True,
    "subtitle": {
        "x": 0.5,
        "y": 0.86,
        "size": 42,
        "outline": 3,
    },
    "trim_start": 0.0,
    "trim_end": 999999.0,
}


if "editor_state" not in st.session_state:
    st.session_state.editor_state = (
        DEFAULT_EDITOR_STATE.copy()
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎬 Myanmar Movie AI Studio"
)

st.caption(
    "Video → Myanmar SRT → Voiceover → Visual Editor → Final MP4"
)


# ============================================================
# STEP 1
# ============================================================

st.markdown(
    "## 1️⃣ Video → မြန်မာ SRT"
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
            bar = st.progress(0.0)

            status.info(
                "Video audio ထုတ်နေသည်..."
            )

            extract_audio(
                video,
                audio,
            )

            bar.progress(0.15)

            status.info(
                "Deepgram က dialogue + "
                "word timestamp ရယူနေသည်..."
            )

            source_segments = (
                deepgram_transcribe(
                    audio
                )
            )

            bar.progress(0.30)

            status.info(
                "Gemini က မြန်မာလို "
                "ဘာသာပြန်နေသည်..."
            )

            segments = build_segments(
                gemini_client(),
                source_segments,
                lambda p, t: (
                    bar.progress(
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
                "editor_segments"
            ] = segments

            bar.progress(1.0)

            status.success(
                f"SRT ပြီးပါပြီ — "
                f"{len(segments)} lines"
            )

    except Exception as e:
        st.error(
            "SRT ထုတ်ရာမှာ "
            "အမှားဖြစ်ပါတယ်။"
        )
        st.exception(e)


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
        height=300,
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


# ============================================================
# STEP 2
# ============================================================

st.markdown("---")

st.markdown(
    "## 2️⃣ SRT → မြန်မာ Voiceover"
)

srt_file = st.file_uploader(
    "📄 SRT တင်ပါ",
    type=["srt"],
    key="srt_upload",
)

voice_name = st.selectbox(
    "🎙️ Voice",
    list(VOICES),
)

style = st.selectbox(
    "🎭 Deep / Style",
    list(VOICE_STYLES),
)

speed = st.slider(
    "⚡ Speed",
    0.70,
    1.30,
    1.00,
    0.05,
)

voice_filename = st.text_input(
    "💾 Voiceover filename",
    value="myanmar_voiceover",
)

make_voice_button = st.button(
    "🗣️ Voiceover ထုတ်မယ်",
    type="primary",
    use_container_width=True,
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

        with tempfile.TemporaryDirectory() as td:

            work = Path(td)

            status = st.empty()
            bar = st.progress(0.0)

            status.info(
                f"SRT timing စစ်နေသည်... "
                f"{len(segments)} lines"
            )

            voiceover = build_voiceover(
                segments,
                VOICES[voice_name],
                style,
                speed,
                work,
                lambda p, t: (
                    bar.progress(
                        min(p, 1.0)
                    ),
                    status.info(t),
                ),
            )

            voice_data = (
                voiceover.read_bytes()
            )

            filename = safe_filename(
                voice_filename
            )

            if not filename.lower().endswith(
                ".m4a"
            ):
                filename += ".m4a"

            st.session_state[
                "voice_bytes"
            ] = voice_data

            st.session_state[
                "voice_name"
            ] = filename

            st.session_state[
                "voice_mime"
            ] = "audio/mp4"

            st.session_state[
                "voice_segments"
            ] = segments

            bar.progress(1.0)

            status.success(
                "Voiceover ပြီးပါပြီ — "
                "SRT timing အတိုင်း "
                "audio ပြုလုပ်ပြီးပါပြီ။"
            )

    except Exception as e:
        st.error(
            "Voiceover ထုတ်ရာမှာ "
            "အမှားဖြစ်ပါတယ်။"
        )
        st.exception(e)


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
# STEP 3 — VISUAL EDITOR
# ============================================================

st.markdown("---")

st.markdown(
    "## 3️⃣ 🎬 Visual Edit Studio"
)

st.caption(
    "Video ကိုမြင်ရင်း Mask / Subtitle / Zoom / Position / "
    "Flip / Ratio ကို တိုက်ရိုက်ချိန်နိုင်ပါတယ်။"
)

editor_video = st.file_uploader(
    "🎥 Step 3 အတွက် Video",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm",
    ],
    key="editor_video",
)

if editor_video:

    if "editor_source_name" not in st.session_state:
        st.session_state[
            "editor_source_name"
        ] = editor_video.name

    if (
        st.session_state.get(
            "editor_source_bytes"
        )
        != editor_video.getvalue()
    ):
        st.session_state[
            "editor_source_bytes"
        ] = editor_video.getvalue()

        st.session_state[
            "editor_state"
        ] = json.loads(
            json.dumps(
                DEFAULT_EDITOR_STATE
            )
        )

        st.session_state.pop(
            "editor_result",
            None,
        )

    try:

        with tempfile.TemporaryDirectory() as td:

            work = Path(td)

            source = (
                work
                / "editor_input.mp4"
            )

            preview = (
                work
                / "editor_preview.mp4"
            )

            source.write_bytes(
                editor_video.getbuffer()
            )

            make_preview_video(
                source,
                preview,
            )

            source_url = video_data_url(
                preview
            )

            segments = (
                st.session_state.get(
                    "voice_segments"
                )
                or st.session_state.get(
                    "editor_segments"
                )
                or []
            )

            if not segments and st.session_state.get(
                "srt_text"
            ):
                try:
                    segments = parse_srt(
                        st.session_state[
                            "srt_text"
                        ]
                    )
                except Exception:
                    segments = []

            editor_result = components.declare_component(
                "visual_video_editor",
                path=str(
                    EDITOR_DIR
                ),
            )(
                video=source_url,
                duration=ffprobe_duration(
                    source
                ),
                subtitles=segments,
                initial_state=st.session_state[
                    "editor_state"
                ],
                key="visual_video_editor",
                default=st.session_state[
                    "editor_state"
                ],
            )

            if (
                isinstance(
                    editor_result,
                    dict,
                )
                and editor_result
            ):
                st.session_state[
                    "editor_state"
                ] = editor_result

            current_state = (
                st.session_state[
                    "editor_state"
                ]
            )

            st.markdown(
                "### 🎧 Audio"
            )

            audio_col1, audio_col2 = (
                st.columns(2)
            )

            with audio_col1:
                use_voice = st.checkbox(
                    "🗣️ Generated Voiceover သုံးမယ်",
                    value=bool(
                        st.session_state.get(
                            "voice_bytes"
                        )
                    ),
                )

            with audio_col2:
                original_volume = st.slider(
                    "Original Audio",
                    0.0,
                    1.0,
                    0.0,
                    0.05,
                )

            music_file = st.file_uploader(
                "🎵 Background Music (optional)",
                type=[
                    "mp3",
                    "wav",
                    "m4a",
                    "aac",
                    "ogg",
                ],
                key="editor_music",
            )

            music_volume = st.slider(
                "🎵 Music Volume",
                0.0,
                1.0,
                0.15,
                0.05,
            )

            voice_volume = st.slider(
                "🗣️ Voice Volume",
                0.5,
                2.0,
                1.0,
                0.05,
            )

            st.markdown(
                "### ✂️ Trim / Output"
            )

            video_duration = (
                ffprobe_duration(
                    source
                )
            )

            trim_col1, trim_col2 = (
                st.columns(2)
            )

            with trim_col1:
                trim_start = st.number_input(
                    "Start (sec)",
                    min_value=0.0,
                    max_value=max(
                        0.0,
                        video_duration
                        - 0.05,
                    ),
                    value=min(
                        float(
                            current_state.get(
                                "trim_start",
                                0.0,
                            )
                        ),
                        max(
                            0.0,
                            video_duration
                            - 0.05,
                        ),
                    ),
                    step=0.1,
                )

            with trim_col2:
                trim_end = st.number_input(
                    "End (sec)",
                    min_value=0.05,
                    max_value=video_duration,
                    value=min(
                        float(
                            current_state.get(
                                "trim_end",
                                video_duration,
                            )
                        ),
                        video_duration,
                    ),
                    step=0.1,
                )

            current_state[
                "trim_start"
            ] = trim_start

            current_state[
                "trim_end"
            ] = max(
                trim_start + 0.05,
                trim_end,
            )

            output_name = st.text_input(
                "💾 Final filename",
                value="Myanmar_Final",
            )

            render_button = st.button(
                "🎬 RENDER FINAL MP4",
                type="primary",
                use_container_width=True,
            )

            if render_button:

                if (
                    trim_end
                    <= trim_start
                ):
                    st.error(
                        "Trim End က "
                        "Trim Start ထက် ကြီးရပါမယ်။"
                    )
                    st.stop()

                try:

                    with tempfile.TemporaryDirectory() as render_td:

                        render_work = Path(
                            render_td
                        )

                        render_video = (
                            render_work
                            / "input.mp4"
                        )

                        render_video.write_bytes(
                            editor_video.getbuffer()
                        )

                        voice_path = None

                        if (
                            use_voice
                            and st.session_state.get(
                                "voice_bytes"
                            )
                        ):
                            voice_path = (
                                render_work
                                / "voice.m4a"
                            )

                            voice_path.write_bytes(
                                st.session_state[
                                    "voice_bytes"
                                ]
                            )

                        music_path = None

                        if music_file:
                            music_path = (
                                render_work
                                / "music"
                            )

                            music_path.write_bytes(
                                music_file.getbuffer()
                            )

                        final_name = safe_filename(
                            output_name
                        )

                        if not final_name.lower().endswith(
                            ".mp4"
                        ):
                            final_name += ".mp4"

                        final_path = (
                            render_work
                            / final_name
                        )

                        status = st.empty()
                        progress = st.progress(
                            0.0
                        )

                        status.info(
                            "🎬 Final video render "
                            "လုပ်နေသည်..."
                        )

                        progress.progress(
                            0.25
                        )

                        render_manual(
                            render_video,
                            final_path,
                            current_state,
                            segments,
                            voice_path=voice_path,
                            music_path=music_path,
                            original_volume=original_volume,
                            voice_volume=voice_volume,
                            music_volume=music_volume,
                        )

                        progress.progress(
                            1.0
                        )

                        result_bytes = (
                            final_path.read_bytes()
                        )

                        st.session_state[
                            "editor_result"
                        ] = result_bytes

                        st.session_state[
                            "editor_result_name"
                        ] = final_name

                        status.success(
                            "✅ Final MP4 "
                            "အောင်မြင်စွာ ပြီးပါပြီ။"
                        )

                except Exception as e:
                    st.error(
                        "Final video render "
                        "မအောင်မြင်ပါ။"
                    )
                    st.exception(e)


if st.session_state.get(
    "editor_result"
):

    st.markdown(
        "### 🎬 Final Video"
    )

    st.video(
        st.session_state[
            "editor_result"
        ]
    )

    st.download_button(
        "⬇️ DOWNLOAD FINAL MP4",
        st.session_state[
            "editor_result"
        ],
