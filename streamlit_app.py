import os
import re
import json
import time
import asyncio
import random
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
import edge_tts
import imageio_ffmpeg
from PIL import Image, ImageFilter, ImageDraw, ImageFont


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Movie Dubbing AI",
    page_icon="🎬",
    layout="wide",
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

VOICES = {
    "သီဟ (အမျိုးသား)": "my-MM-ThihaNeural",
    "နီလာ (အမျိုးသမီး)": "my-MM-NilarNeural",
}

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

TEMP_WORDS = (
    "503",
    "500",
    "502",
    "504",
    "429",
    "timeout",
    "unavailable",
    "overloaded",
    "high demand",
    "resource exhausted",
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


def duration(path):
    r = run_cmd(
        [FFMPEG, "-hide_banner", "-i", str(path)],
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
            "Original audio ထုတ်မရပါ။\n"
            + (r.stderr or "")
        )


def safe_name(name, default="final_movie.mp4"):
    name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        (name or "").strip(),
    )

    if not name:
        name = default

    if not name.lower().endswith(".mp4"):
        name += ".mp4"

    return name


# ============================================================
# GEMINI
# ============================================================

@st.cache_resource
def get_client():
    key = st.secrets.get(
        "GEMINI_API_KEY",
        os.getenv("GEMINI_API_KEY", ""),
    )

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return genai.Client(api_key=key)


def temporary_error(e):
    text = str(e).lower()
    return any(x.lower() in text for x in TEMP_WORDS)


def model_list(c):
    try:
        names = []

        for m in c.models.list():
            n = getattr(m, "name", "")

            if n:
                names.append(n.replace("models/", ""))

        usable = [
            m for m in MODELS
            if m in names
        ]

        return usable or MODELS

    except Exception:
        return MODELS


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


def normalize(items, dur):
    out = []

    if not isinstance(items, list):
        raise RuntimeError(
            "AI response က JSON list မဟုတ်ပါ။"
        )

    for item in items:
        if not isinstance(item, dict):
            continue

        try:
            start = float(item.get("start", 0))
            end = float(item.get("end", 0))
        except Exception:
            continue

        text = str(
            item.get("burmese", "")
        ).strip()

        start = max(0, min(start, dur))
        end = max(0, min(end, dur))

        if text and end - start >= 0.20:
            out.append(
                {
                    "start": start,
                    "end": end,
                    "burmese": text,
                }
            )

    out.sort(key=lambda x: x["start"])

    cleaned = []

    for item in out:
        if (
            cleaned
            and abs(
                item["start"]
                - cleaned[-1]["start"]
            ) < 0.05
            and item["burmese"]
            == cleaned[-1]["burmese"]
        ):
            cleaned[-1]["end"] = max(
                cleaned[-1]["end"],
                item["end"],
            )
        else:
            cleaned.append(item)

    return cleaned


def wait_file(c, f):
    name = getattr(f, "name", None)

    if not name:
        return f

    for _ in range(90):
        try:
            cur = c.files.get(name=name)

            state = str(
                getattr(
                    getattr(cur, "state", None),
                    "name",
                    getattr(cur, "state", ""),
                )
            ).upper()

            if "PROCESSING" not in state:
                return cur

        except Exception:
            return f

        time.sleep(2)

    return f


def analyze(c, audio, dur, status):
    uploaded = wait_file(
        c,
        c.files.upload(file=str(audio)),
    )

    prompt = f"""
You are a professional movie dubbing editor.

Analyze the uploaded movie audio and find ALL meaningful spoken dialogue.

Requirements:

1. Do not invent dialogue.
2. Keep chronological order.
3. Return accurate approximate start/end timestamps in seconds.
4. Translate every dialogue into natural conversational Burmese.
5. Preserve meaning, emotion, names and relationships.
6. Exclude music and sound effects.
7. Do not merge unrelated dialogue.
8. Keep Burmese lines concise enough for movie subtitles.
9. Each subtitle should normally be short.
10. Prefer natural spoken Burmese rather than literal translation.

Return ONLY valid JSON.

Audio duration:
{dur:.2f} seconds.

Format:

[
  {{
    "start": 10.25,
    "end": 13.80,
    "burmese": "မြန်မာဘာသာပြန်"
  }}
]
"""

    errors = []

    for model in model_list(c):

        status(
            f"AI model: {model}"
        )

        for attempt in range(2):

            try:
                resp = c.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_uri(
                            file_uri=uploaded.uri,
                            mime_type=(
                                getattr(
                                    uploaded,
                                    "mime_type",
                                    None,
                                )
                                or "audio/wav"
                            ),
                        ),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.15
                    ),
                )

                raw = getattr(
                    resp,
                    "text",
                    "",
                )

                seg = normalize(
                    json.loads(
                        clean_json(raw)
                    ),
                    dur,
                )

                if seg:
                    return seg, model

                raise RuntimeError(
                    "Dialogue မတွေ့ပါ။"
                )

            except Exception as e:

                errors.append(
                    f"{model} "
                    f"attempt {attempt + 1}: "
                    f"{e}"
                )

                if (
                    attempt == 0
                    and temporary_error(e)
                ):
                    wait = (
                        4
                        + random.uniform(0, 2)
                    )

                    status(
                        f"{model} busy — "
                        f"{wait:.1f}s retry"
                    )

                    time.sleep(wait)

                else:
                    break

    raise RuntimeError(
        "Gemini model အားလုံးနဲ့ "
        "မအောင်မြင်ပါ။\n"
        + "\n".join(errors[-8:])
    )


# ============================================================
# EDGE TTS
# ============================================================

async def tts_async(
    text,
    voice,
    out,
):
    await edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
        volume="+0%",
    ).save(str(out))


def tts(text, voice, out):
    asyncio.run(
        tts_async(
            text,
            voice,
            out,
        )
    )

    if (
        not out.exists()
        or out.stat().st_size < 1000
    ):
        raise RuntimeError(
            "TTS file မထွက်ပါ။"
        )


def atempo_filter(speed):
    speed = max(
        0.5,
        min(float(speed), 3.0),
    )

    filters = []

    while speed > 2:
        filters.append("atempo=2")
        speed /= 2

    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5

    filters.append(
        f"atempo={speed:.6f}"
    )

    return ",".join(filters)


def fit_tts(
    src,
    out,
    slot,
):
    d = duration(src)

    speed = max(
        0.5,
        min(
            d / max(slot, 0.25),
            3.0,
        ),
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
            atempo_filter(speed),
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(out),
        ],
        180,
    )

    if (
        r.returncode
        or not out.exists()
    ):
        raise RuntimeError(
            "TTS timing ပြင်မရပါ။\n"
            + (r.stderr or "")
        )


def validate_audio(path):
    r = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        300,
    )

    return r.returncode == 0


def make_burmese_audio(
    segments,
    voice,
    dur,
    work,
    progress,
):
    files = []
    total = len(segments)

    for i, item in enumerate(
        segments,
        1,
    ):

        progress(
            (i - 1)
            / max(total, 1),
            f"TTS {i}/{total}",
        )

        raw = (
            work
            / f"tts_{i:04d}.mp3"
        )

        fitted = (
            work
            / f"fit_{i:04d}.m4a"
        )

        tts(
            item["burmese"],
            voice,
            raw,
        )

        fit_tts(
            raw,
            fitted,
            max(
                0.25,
                item["end"]
                - item["start"],
            ),
        )

        files.append(
            (
                item["start"],
                fitted,
            )
        )

    if not files:
        raise RuntimeError(
            "Burmese dialogue မရှိပါ။"
        )

    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    for _, f in files:
        cmd += [
            "-i",
            str(f),
        ]

    filters = []
    labels = []

    for i, (start, _) in enumerate(
        files
    ):

        ms = int(
            round(start * 1000)
        )

        label = f"a{i}"

        filters.append(
            f"[{i}:a]"
            f"aresample=48000,"
            f"adelay={ms}|{ms},"
            f"apad[{label}]"
        )

        labels.append(
            f"[{label}]"
        )

    filters.append(
        "".join(labels)
        + f"amix="
        f"inputs={len(labels)}:"
        f"duration=longest:"
        f"dropout_transition=0"
        f"[mix]"
    )

    out = (
        work
        / "burmese_audio.m4a"
    )

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[mix]",
        "-t",
        f"{dur:.3f}",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
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
        or not validate_audio(out)
    ):
        raise RuntimeError(
            "Burmese audio "
            "mixing/validation "
            "မအောင်မြင်ပါ။\n"
            + (r.stderr or "")
        )

    progress(
        1.0,
        "Burmese audio OK",
    )

    return out


# ============================================================
# SIMPLE FINAL EXPORT
# ============================================================

def validate_video(path):
    if (
        not path.exists()
        or path.stat().st_size < 10000
    ):
        return (
            False,
            "Final video file မမှန်ပါ။",
        )

    v = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        300,
    )

    if v.returncode:
        return (
            False,
            "Video decode မအောင်မြင်ပါ။",
        )

    a = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        300,
    )

    if a.returncode:
        return (
            False,
            "Final video ထဲမှာ "
            "Audio track မရှိပါ "
            "သို့မဟုတ် decode မရပါ။",
        )

    return True, "OK"


def export_video(
    video,
    audio,
    out,
):
    d = duration(video)

    if d <= 0:
        raise RuntimeError(
            "Video duration မမှန်ပါ။"
        )

    r = run_cmd(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            f"{d:.3f}",
            "-movflags",
            "+faststart",
            str(out),
        ],
        1800,
    )

    if r.returncode:

        r = run_cmd(
            [
                FFMPEG,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-i",
                str(audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-t",
                f"{d:.3f}",
                "-movflags",
                "+faststart",
                str(out),
            ],
            3600,
        )

    if r.returncode:
        raise RuntimeError(
            "Final video export "
            "မအောင်မြင်ပါ။\n"
            + (r.stderr or "")
        )

    ok, msg = validate_video(out)

    if not ok:
        raise RuntimeError(
            "Final video validation "
            f"မအောင်မြင်ပါ။\n{msg}"
        )


# ============================================================
# STEP 1 + STEP 2
# ============================================================

st.title("🎬 Movie Dubbing AI")

st.caption(
    "Video → AI dialogue → "
    "မြန်မာ SRT → Voiceover → "
    "Visual Edit → Final MP4"
)


if "final_bytes" not in st.session_state:
    st.session_state.final_bytes = None

if "final_name" not in st.session_state:
    st.session_state.final_name = (
        "dubbed_video.mp4"
    )

if "srt_text" not in st.session_state:
    st.session_state.srt_text = ""

if "voice_bytes" not in st.session_state:
    st.session_state.voice_bytes = None


# ------------------------------------------------------------
# STEP 1
# ------------------------------------------------------------

st.markdown("---")
st.header("1️⃣ 🎙️ Video → မြန်မာ SRT")

uploaded = st.file_uploader(
    "🎥 Video တင်ပါ",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm",
    ],
    key="step1_video",
)

voice_name = st.selectbox(
    "🎙️ Voice",
    list(VOICES),
    key="step1_voice",
)

start = st.button(
    "🚀 Start AI Dubbing",
    type="primary",
    use_container_width=True,
    key="start_ai",
)


if start:

    st.session_state.final_bytes = None

    if not uploaded:
        st.error(
            "Video တစ်ခုအရင်တင်ပါ။"
        )
        st.stop()

    try:

        with tempfile.TemporaryDirectory() as td:

            work = Path(td)

            video = (
                work / "input_video"
            )

            video.write_bytes(
                uploaded.getbuffer()
            )

            dur = duration(video)

            status = st.empty()
            bar = st.progress(0.0)

            status.info(
                "1/4 Video စစ်နေသည်..."
            )

            audio = (
                work
                / "original_audio.wav"
            )

            extract_audio(
                video,
                audio,
            )

            bar.progress(0.10)

            status.info(
                "2/4 AI dialogue "
                "နားထောင်/ဘာသာပြန်နေသည်..."
            )

            seg, model = analyze(
                get_client(),
                audio,
                dur,
                status.info,
            )

            bar.progress(0.45)

            st.success(
                f"AI model: {model} • "
                f"Dialogue: {len(seg)} lines"
            )

            # SRT
            srt_lines = []

            for i, item in enumerate(
                seg,
                1,
            ):

                def fmt_srt(v):
                    v = max(
                        0,
                        float(v),
                    )

                    ms = int(
                        round(
                            (v - int(v))
                            * 1000
                        )
                    )

                    sec = int(v)

                    if ms >= 1000:
                        sec += 1
                        ms = 0

                    h = sec // 3600
                    sec %= 3600

                    m = sec // 60
                    sec %= 60

                    return (
                        f"{h:02d}:"
                        f"{m:02d}:"
                        f"{sec:02d},"
                        f"{ms:03d}"
                    )

                srt_lines += [
                    str(i),
                    (
                        f"{fmt_srt(item['start'])}"
                        f" --> "
                        f"{fmt_srt(item['end'])}"
                    ),
                    item["burmese"],
                    "",
                ]

            srt_text = "\n".join(
                srt_lines
            )

            st.session_state.srt_text = (
                srt_text
            )

            st.session_state.srt_name = (
                "myanmar_subtitles.srt"
            )

            st.success(
                "✅ မြန်မာ SRT ပြီးပါပြီ။"
            )

            bar.progress(0.60)

            status.info(
                "3/4 Burmese Voice "
                "ထုတ်နေသည်..."
            )

            def voice_progress(
                p,
                text,
            ):
                bar.progress(
                    0.60 + 0.35 * p
                )
                status.info(text)

            voice_file = (
                make_burmese_audio(
                    seg,
                    VOICES[voice_name],
                    dur,
                    work,
                    voice_progress,
                )
            )

            st.session_state.voice_bytes = (
                voice_file.read_bytes()
            )

            st.session_state.voice_name = (
                "myanmar_voiceover.m4a"
            )

            st.session_state.voice_mime = (
                "audio/mp4"
            )

            st.session_state.voice_segments = (
                seg
            )

            bar.progress(1.0)

            status.success(
                "✅ SRT + Voiceover ပြီးပါပြီ။"
            )

    except Exception as e:
        st.exception(e)


if st.session_state.srt_text:

    st.subheader(
        "📄 မြန်မာ SRT Preview"
    )

    st.text_area(
        "မြန်မာ SRT",
        st.session_state.srt_text,
        height=300,
        key="srt_preview",
    )

    st.download_button(
        "⬇️ Download Myanmar SRT",
        st.session_state.srt_text.encode(
            "utf-8-sig"
        ),
        st.session_state.get(
            "srt_name",
            "myanmar.srt",
        ),
        "application/x-subrip",
        use_container_width=True,
        key="download_srt",
    )


# ------------------------------------------------------------
# STEP 2
# ------------------------------------------------------------

st.markdown("---")
st.header(
    "2️⃣ 🗣️ SRT → မြန်မာ Voiceover"
)

srt_file = st.file_uploader(
    "📄 SRT တင်ပါ",
    type=["srt"],
    key="srt_upload",
)

voice_name_2 = st.selectbox(
    "🎙️ Voice",
    list(VOICES),
    key="voice_step2",
)

style = st.selectbox(
    "🎭 Voice Style",
    [
        "ပုံမှန်",
        "နက်နက် (Deep)",
        "ပျော့ပျောင်း",
        "တက်ကြွ",
    ],
    key="voice_style",
)

speed = st.slider(
    "⚡ Speed",
    0.70,
    1.30,
    1.00,
    0.05,
    key="voice_speed",
)

voice_filename = st.text_input(
    "📁 Voiceover Filename",
    "myanmar_voiceover.m4a",
    key="voice_filename",
)

make_voice_button = st.button(
    "🗣️ Voiceover ထုတ်မယ်",
    type="primary",
    use_container_width=True,
    key="make_voice",
)


def parse_srt(text):
    rows = []

    blocks = re.split(
        r"\n\s*\n",
        (text or "").strip(),
    )

    for block in blocks:

        lines = [
            x.strip("\ufeff")
            for x in block.splitlines()
            if x.strip()
        ]

        if len(lines) < 3:
            continue

        m = re.search(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
            r"\s*-->\s*"
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})",
            lines[1],
        )

        if not m:
            continue

        a = [
            int(m.group(i))
            for i in range(1, 5)
        ]

        b = [
            int(m.group(i))
            for i in range(5, 9)
        ]

        start = (
            a[0] * 3600
            + a[1] * 60
            + a[2]
            + a[3] / 1000
        )

        end = (
            b[0] * 3600
            + b[1] * 60
            + b[2]
            + b[3] / 1000
        )

        text_value = " ".join(
            lines[2:]
        ).strip()

        if (
            text_value
            and end > start
        ):
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "burmese": text_value,
                }
            )

    return rows


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

    else:
        st.error(
            "SRT ဖိုင်တင်ပါ "
            "(သို့) Step 1 မှာ SRT "
            "အရင်ထုတ်ပါ။"
        )
        st.stop()

    try:

        segments = parse_srt(
            source_srt
        )

        if not segments:
            raise RuntimeError(
                "SRT ထဲမှာ valid subtitle "
                "မတွေ့ပါ။"
            )

        with tempfile.TemporaryDirectory() as td:

            work = Path(td)

            status = st.empty()
            bar = st.progress(0.0)

            status.info(
                f"SRT timing စစ်နေသည်... "
                f"{len(segments)} lines"
            )

            voice_style_values = {
                "ပုံမှန်": {
                    "rate": "+0%",
                    "pitch": "+0Hz",
                },
                "နက်နက် (Deep)": {
                    "rate": "-5%",
                    "pitch": "-12Hz",
                },
                "ပျော့ပျောင်း": {
                    "rate": "-3%",
                    "pitch": "+5Hz",
                },
                "တက်ကြွ": {
                    "rate": "+8%",
                    "pitch": "+2Hz",
                },
            }

            style_cfg = (
                voice_style_values[
                    style
                ]
            )

            clips = []

            total = len(
                segments
            )

            for i, item in enumerate(
                segments,
                1,
            ):

                raw = (
                    work
                    / f"voice_{i:04d}.mp3"
                )

                fitted = (
                    work
                    / f"fit_{i:04d}.m4a"
                )

                rate = (
                    style_cfg["rate"]
                )

                pitch = (
                    style_cfg["pitch"]
                )

                async def make_voice(
                    text,
                    output,
                    voice,
                    rate,
                    pitch,
                ):
                    await edge_tts.Communicate(
                        text=text,
                        voice=voice,
                        rate=rate,
                        pitch=pitch,
                        volume="+0%",
                    ).save(
                        str(output)
                    )

                asyncio.run(
                    make_voice(
                        item["burmese"],
                        raw,
                        VOICES[
                            voice_name_2
                        ],
                        rate,
                        pitch,
                    )
                )

                if (
                    not raw.exists()
                    or raw.stat().st_size
                    < 1000
                ):
                    raise RuntimeError(
                        "Voice file မထွက်ပါ။"
                    )

                slot = max(
                    0.25,
                    item["end"]
                    - item["start"],
                )

                fit_tts(
                    raw,
                    fitted,
                    slot,
                )

                clips.append(
                    (
                        item["start"],
                        fitted,
                    )
                )

                bar.progress(
                    0.1
                    + 0.8
                    * i
                    / total
                )

                status.info(
                    f"Voice {i}/{total}"
                )

            out = (
                work
                / safe_name(
                    voice_filename,
                    "myanmar_voiceover.m4a",
                ).replace(
                    ".mp4",
                    ".m4a",
                )
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
                start_time,
                _,
            ) in enumerate(clips):

                ms = int(
                    round(
                        start_time
                        * 1000
                    )
                )

                label = f"v{i}"

                filters.append(
                    f"[{i}:a]"
                    f"aresample=48000,"
                    f"adelay={ms}|{ms}"
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
                f"loudnorm=I=-16:"
                f"TP=-1.5:"
                f"LRA=11,"
                f"alimiter=limit=0.95"
                f"[out]"
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
                or out.stat().st_size
                < 5000
            ):
                raise RuntimeError(
                    "Voiceover file "
                    "မထုတ်နိုင်ပါ။\n"
                    + (r.stderr or "")
                )

            data = out.read_bytes()

            st.session_state.voice_bytes = (
                data
            )

            st.session_state.voice_name = (
                safe_name(
                    voice_filename,
                    "myanmar_voiceover.mp4",
                ).replace(
                    ".mp4",
                    ".m4a",
                )
            )

            st.session_state.voice_segments = (
                segments
            )

            bar.progress(1.0)

            status.success(
                "✅ Voiceover ပြီးပါပြီ။"
            )

    except Exception as e:
        st.exception(e)


if st.session_state.get(
    "voice_bytes"
):

    st.subheader(
        "🎧 Voiceover Preview"
    )

    st.audio(
        st.session_state.voice_bytes,
        format="audio/mp4",
    )

    st.download_button(
        "⬇️ Download Voiceover",
        st.session_state.voice_bytes,
        st.session_state.get(
            "voice_name",
            "myanmar_voiceover.m4a",
        ),
        "audio/mp4",
        use_container_width=True,
        key="download_voice",
    )


# ============================================================
# STEP 3 — VISUAL EDITOR
# ============================================================

st.markdown("---")

st.header(
    "3️⃣ 🎬 Manual Visual Edit Studio"
)

st.caption(
    "Video ကိုမြင်ရင်း ချိန် → "
    "Preview ပြောင်းတာကိုချက်ချင်းကြည့် → "
    "အဆင်ပြေမှ Final Render"
)


# ============================================================
# VISUAL EDIT HELPERS
# ============================================================

def fmt_srt_time(v):
    v = max(
        0.0,
        float(v),
    )

    sec = int(v)

    ms = int(
        round(
            (v - sec)
            * 1000
        )
    )

    if ms >= 1000:
        sec += 1
        ms = 0

    h = sec // 3600

    sec %= 3600

    m = sec // 60

    sec %= 60

    return (
        f"{h:02d}:"
        f"{m:02d}:"
        f"{sec:02d},"
        f"{ms:03d}"
    )


def get_font(size):

    candidates = [
        "/usr/share/fonts/truetype/noto/"
        "NotoSansMyanmar-Regular.ttf",

        "/usr/share/fonts/opentype/noto/"
        "NotoSansMyanmar-Regular.ttf",

        "/usr/share/fonts/truetype/noto/"
        "NotoSansMyanmarUI-Regular.ttf",
    ]

    for path in candidates:

        p = Path(path)

        if p.exists():

            try:
                return ImageFont.truetype(
                    str(p),
                    size=size,
                )
            except Exception:
                pass

    return ImageFont.load_default()


def fit_canvas(
    image,
    ratio,
):
    w, h = image.size

    ratios = {
        "Original": w / h,
        "9:16": 9 / 16,
        "16:9": 16 / 9,
        "1:1": 1.0,
        "4:5": 4 / 5,
    }

    target = ratios.get(
        ratio,
        w / h,
    )

    if ratio == "Original":
        return image.copy()

    current = w / h

    if current > target:

        new_w = int(
            h * target
        )

        left = (
            w - new_w
        ) // 2

        return image.crop(
            (
                left,
                0,
                left + new_w,
                h,
            )
        )

    new_h = int(
        w / target
    )

    top = (
        h - new_h
    ) // 2

    return image.crop(
        (
            0,
            top,
            w,
            top + new_h,
        )
    )


def preview_frame(
    video_path,
    t,
):
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False,
    ) as f:
        out = Path(f.name)

    r = run_cmd(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0, t):.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=960:-2",
            "-q:v",
            "3",
            str(out),
        ],
        120,
    )

    if (
        r.returncode
        or not out.exists()
    ):
        raise RuntimeError(
            "Preview frame "
            "ထုတ်မရပါ။\n"
            + (r.stderr or "")
        )

    try:
        return Image.open(
            out
        ).convert("RGB")
    finally:
        try:
            out.unlink()
        except Exception:
            pass


def draw_preview(
    image,
    ratio,
    zoom,
    px,
    py,
    blur,
    mask,
    subtitle,
    subtitle_pos,
    subtitle_size,
    subtitle_outline,
):
    """
    Preview-only renderer.
    User sees the result before Final Render.
    """

    base = fit_canvas(
        image,
        ratio,
    )

    canvas_w, canvas_h = (
        base.size
    )

    # ----------------------------------------
    # BACKGROUND
    # ----------------------------------------

    if (
        blur
        and ratio != "Original"
    ):
        background = base.filter(
            ImageFilter.GaussianBlur(
                radius=18
            )
        )
    else:
        background = base.copy()

    # ----------------------------------------
    # FOREGROUND VIDEO
    # ----------------------------------------

    fg = image.copy()

    base_scale = min(
        canvas_w / fg.width,
        canvas_h / fg.height,
    )

    final_scale = (
        base_scale
        * float(zoom)
    )

    new_w = max(
        1,
        int(
            fg.width
            * final_scale
        ),
    )

    new_h = max(
        1,
        int(
            fg.height
            * final_scale
        ),
    )

    fg = fg.resize(
        (
            new_w,
            new_h,
        ),
        Image.Resampling.LANCZOS,
    )

    x = int(
        (canvas_w - new_w) / 2
        + (
            float(px)
            / 100.0
        )
        * canvas_w
    )

    y = int(
        (canvas_h - new_h) / 2
        + (
            float(py)
            / 100.0
        )
        * canvas_h
    )

    # ----------------------------------------
    # MASK
    # ----------------------------------------

    if mask != "None":

        rgba = Image.new(
            "RGBA",
            fg.size,
            (0, 0, 0, 0),
        )

        alpha = Image.new(
            "L",
            fg.size,
            0,
        )

        draw = ImageDraw.Draw(
            alpha
        )

        if mask == "Circle":

            draw.ellipse(
                (
                    0,
                    0,
                    fg.width,
                    fg.height,
                ),
                fill=255,
            )

        elif mask == "Rectangle":

            radius = max(
                10,
                min(
                    fg.width,
                    fg.height,
                )
                // 12,
            )

            draw.rounded_rectangle(
                (
                    0,
                    0,
                    fg.width,
                    fg.height,
                ),
                radius=radius,
                fill=255,
            )

        rgba.paste(
            fg,
            (0, 0),
            alpha,
        )

        background = (
            background.convert(
                "RGBA"
            )
        )

        background.alpha_composite(
            rgba,
            (
                x,
                y,
            ),
        )

        canvas = background.convert(
            "RGB"
        )

    else:

        canvas = background.convert(
            "RGB"
        )

        canvas.paste(
            fg,
            (
                x,
                y,
            ),
        )

    # ----------------------------------------
    # SAFE AREA BORDER
    # ----------------------------------------

    overlay = Image.new(
        "RGBA",
        canvas.size,
        (0, 0, 0, 0),
    )

    od = ImageDraw.Draw(
        overlay
    )

    od.rectangle(
        (
            2,
            2,
            canvas_w - 3,
            canvas_h - 3,
        ),
        outline=(255, 255, 255, 150),
        width=2,
    )

    canvas = Image.alpha_composite(
        canvas.convert("RGBA"),
        overlay,
    )

    # ----------------------------------------
    # SUBTITLE
    # ----------------------------------------

    if subtitle:

        font = get_font(
            int(subtitle_size)
        )

        max_width = int(
            canvas_w * 0.88
        )

        words = subtitle.split()

        lines = []
        current = ""

        for word in words:

            test = (
                current
                + " "
                + word
            ).strip()

            bbox = font.getbbox(
                test
            )

            if (
                bbox[2] - bbox[0]
                <= max_width
                or not current
            ):
                current = test
            else:
                lines.append(
                    current
                )
                current = word

        if current:
            lines.append(
                current
            )

        line_height = max(
            int(
                subtitle_size
                * 1.4
            ),
            20,
        )

        total_height = (
            line_height
            * len(lines)
        )

        if subtitle_pos == "Top":
            y_text = int(
                canvas_h * 0.08
            )

        elif subtitle_pos == "Middle":
            y_text = int(
                (
                    canvas_h
                    - total_height
                )
                / 2
            )

        else:
            y_text = int(
                canvas_h * 0.80
                - total_height / 2
            )

        draw = ImageDraw.Draw(
            canvas
        )

        for line in lines:

            bbox = draw.textbbox(
                (
                    0,
                    0,
                ),
                line,
                font=font,
                stroke_width=int(
                    subtitle_outline
                ),
            )

            text_w = (
                bbox[2]
                - bbox[0]
            )

            x_text = max(
                4,
                (
                    canvas_w
                    - text_w
                )
                // 2,
            )

            draw.text(
                (
                    x_text,
                    y_text,
                ),
                line,
                font=font,
                fill="white",
                stroke_width=int(
                    subtitle_outline
                ),
                stroke_fill="black",
            )

            y_text += line_height

    return canvas.convert(
        "RGB"
    )


def has_audio(path):
    r = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        60,
    )

    return r.returncode == 0


# ============================================================
# FINAL MANUAL RENDER
# ============================================================

def render_manual(
    video,
    voice,
    srt_text,
    trim_start,
    trim_end,
    ratio,
    flip_h,
    flip_v,
    zoom,
    px,
    py,
    blur,
    mask,
    subtitle_size,
    subtitle_pos,
    subtitle_outline,
    music,
    original_volume,
    voice_volume,
    music_volume,
    quality,
    out,
):

    final_duration = max(
        0.05,
        trim_end - trim_start,
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    vf = [
        (
            f"trim="
            f"start={trim_start:.3f}:"
            f"end={trim_end:.3f}"
        ),
        "setpts=PTS-STARTPTS",
    ]

    if flip_h:
        vf.append("hflip")

    if flip_v:
        vf.append("vflip")

    if ratio != "Original":

        target = {
            "9:16": "9/16",
            "16:9": "16/9",
            "1:1": "1",
            "4:5": "4/5",
        }[ratio]

        # Make a canvas according to selected ratio.
        vf.append(
            "scale="
            f"iw*{zoom:.4f}:"
            f"ih*{zoom:.4f}:"
            "force_original_aspect_ratio=decrease"
        )

        vf.append(
            "pad="
            f"ceil(iw/{target}/2)*2:"
            "ceil(ih/2)*2:"
            f"(ow-iw)/2+"
            f"({px / 100:.4f})*ow:"
            f"(oh-ih)/2+"
            f"({py / 100:.4f})*oh"
        )

    else:

        vf.append(
            f"scale="
            f"iw*{zoom:.4f}:"
            f"ih*{zoom:.4f}"
        )

    # --------------------------------------------------------
    # BACKGROUND BLUR
    # --------------------------------------------------------

    if blur:
        vf.append(
            "boxblur=6:2"
        )

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    # Keep mask rendering stable.
    # Circle = alpha outside circle.
    if mask == "Circle":

        vf.append(
            "format=rgba,"
            "geq="
            "r=r:g=g:b=b:"
            "a=if("
            "gt("
            "pow(X-W/2,2)+"
            "pow(Y-H/2,2),"
            "pow(min(W,H)/2,2)"
            "),"
            "0,"
            "255"
            "),"
            "format=yuv420p"
        )

    elif mask == "Rectangle":

        vf.append(
            "format=rgba"
        )

    # --------------------------------------------------------
    # SUBTITLE
    # --------------------------------------------------------

    if srt_text:

        rows = parse_srt(
            srt_text
        )

        kept = []

        for row in rows:

            a = max(
                0,
                row["start"]
                - trim_start,
            )

            b = min(
                final_duration,
                row["end"]
                - trim_start,
            )

            if b - a >= 0.05:

                kept.append(
                    (
                        a,
                        b,
                        row["burmese"],
                    )
                )

        if kept:

            srt_tmp = Path(
                tempfile.mkstemp(
                    suffix=".srt"
                )[1]
            )

            lines = []

            for i, (
                a,
                b,
                text_value,
            ) in enumerate(
                kept,
                1,
            ):

                lines += [
                    str(i),
                    (
                        f"{fmt_srt_time(a)}"
                        f" --> "
                        f"{fmt_srt_time(b)}"
                    ),
                    text_value,
                    "",
                ]

            srt_tmp.write_text(
                "\n".join(lines),
                encoding="utf-8",
            )

            # ------------------------------------------------
            # ASS subtitle
            # ------------------------------------------------

            ass_file = (
                srt_tmp.with_suffix(
                    ".ass"
                )
            )

            alignment = {
                "Top": 8,
                "Middle": 5,
                "Bottom": 2,
            }[subtitle_pos]

            ass_lines = [
                "[Script Info]",
                "ScriptType: v4.00+",
                "PlayResX: 1920",
                "PlayResY: 1080",
                "",
                "[V4+ Styles]",
                (
                    "Format: Name, Fontname, "
                    "Fontsize, PrimaryColour, "
                    "SecondaryColour, "
                    "OutlineColour, BackColour, "
                    "Bold, Italic, Underline, "
                    "StrikeOut, ScaleX, ScaleY, "
                    "Spacing, Angle, BorderStyle, "
                    "Outline, Shadow, Alignment, "
                    "MarginL, MarginR, MarginV, "
                    "Encoding"
                ),
                (
                    "Style: Default,"
                    "Noto Sans Myanmar,"
                    f"{int(subtitle_size * 1.35)},"
                    "&H00FFFFFF,"
                    "&H00FFFFFF,"
                    "&H00000000,"
                    "&H80000000,"
                    "0,0,0,0,100,100,0,0,1,"
                    f"{int(subtitle_outline * 2)},"
                    "0,"
                    f"{alignment},"
                    "60,60,120,1"
                ),
                "",
                "[Events]",
                (
                    "Format: Layer, Start, End, "
                    "Style, Name, MarginL, "
                    "MarginR, MarginV, Effect, Text"
                ),
            ]

            def ass_time(v):

                cs = int(
                    round(
                        v * 100
                    )
                )

                hh = (
                    cs
                    // 360000
                )

                cs %= 360000

                mm = (
                    cs
                    // 6000
                )

                cs %= 6000

                ss = (
                    cs
                    // 100
                )

                cc = (
                    cs
                    % 100
                )

                return (
                    f"{hh}:"
                    f"{mm:02d}:"
                    f"{ss:02d}."
                    f"{cc:02d}"
                )

            for (
                a,
                b,
                text_value,
            ) in kept:

                safe_text = (
                    text_value
                    .replace(
                        "{",
                        r"\{",
                    )
                    .replace(
                        "}",
                        r"\}",
                    )
                )

                ass_lines.append(
                    (
                        "Dialogue: 0,"
                        f"{ass_time(a)},"
                        f"{ass_time(b)},"
                        "Default,,0,0,0,,"
                        f"{safe_text}"
                    )
                )

            ass_file.write_text(
                "\n".join(
                    ass_lines
                ),
                encoding="utf-8",
            )

            ass_path = str(
                ass_file
            ).replace(
                "\\",
                "/",
            )

            vf.append(
                "subtitles="
                "'"
                + ass_path.replace(
                    "'",
                    "\\'",
                )
                + "'"
            )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
    ]

    input_count = 1

    voice_index = None
    music_index = None

    if (
        voice
        and Path(voice).exists()
    ):

        voice_index = input_count

        cmd += [
            "-i",
            str(voice),
        ]

        input_count += 1

    if (
        music
        and Path(music).exists()
    ):

        music_index = input_count

        cmd += [
            "-stream_loop",
            "-1",
            "-i",
            str(music),
        ]

        input_count += 1

    # --------------------------------------------------------
    # VIDEO FILTER
    # --------------------------------------------------------

    filter_complex = (
        "[0:v]"
        + ",".join(vf)
        + "[v]"
    )

    audio_labels = []

    # Original audio
    if (
        has_audio(video)
        and original_volume > 0
    ):

        filter_complex += (
            ";[0:a]"
            f"atrim="
            f"start={trim_start:.3f}:"
            f"end={trim_end:.3f},"
            "asetpts=PTS-STARTPTS,"
            f"volume="
            f"{original_volume / 100:.3f}"
            "[oa]"
        )

        audio_labels.append(
            "[oa]"
        )

    # Voiceover
    if (
        voice_index is not None
        and voice_volume > 0
    ):

        filter_complex += (
            f";[{voice_index}:a]"
            f"atrim=0:"
            f"{final_duration:.3f},"
            "asetpts=PTS-STARTPTS,"
            f"volume="
            f"{voice_volume / 100:.3f}"
            "[va]"
        )

        audio_labels.append(
            "[va]"
        )

    # Background music
    if (
        music_index is not None
        and music_volume > 0
    ):

        filter_complex += (
            f";[{music_index}:a]"
            f"atrim=0:"
            f"{final_duration:.3f},"
            "asetpts=PTS-STARTPTS,"
            f"volume="
            f"{music_volume / 100:.3f}"
            "[ma]"
        )

        audio_labels.append(
            "[ma]"
        )

    # --------------------------------------------------------
    # MIX AUDIO
    # --------------------------------------------------------

    audio_map = None

    if audio_labels:

        filter_complex += (
            ";"
            + "".join(
                audio_labels
            )
            + f"amix="
            f"inputs={len(audio_labels)}:"
            f"duration=longest:"
            "dropout_transition=0,"
            "loudnorm="
            "I=-16:"
            "TP=-1.5:"
            "LRA=11,"
            "alimiter="
            "limit=0.95"
            "[a]"
        )

        audio_map = "[a]"

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
    ]

    if audio_map:
        cmd += [
            "-map",
            audio_map,
        ]

    crf = {
        "High": "18",
        "Good": "21",
        "Small": "26",
    }[quality]

    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        crf,
        "-pix_fmt",
        "yuv420p",
    ]

    if audio_map:

        cmd += [
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
        ]

    else:

        cmd += [
            "-an",
        ]

    cmd += [
        "-t",
        f"{final_duration:.3f}",
        "-movflags",
        "+faststart",
        str(out),
    ]

    result = run_cmd(
        cmd,
        3600,
    )

    if (
        result.returncode
        or not out.exists()
        or out.stat().st_size < 10000
    ):
        raise RuntimeError(
            "Final Render "
            "မအောင်မြင်ပါ။\n"
            + (
                result.stderr
                or ""
            )
        )

    return out


# ============================================================
# STEP 3 UI
# ============================================================

edit_video = st.file_uploader(
    "🎥 Edit လုပ်မယ့် Video",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm",
    ],
    key="edit_video_upload",
)


if edit_video:

    if (
        "edit_video_bytes"
        not in st.session_state
        or st.session_state.get(
            "edit_video_name"
        )
        != edit_video.name
    ):

        st.session_state.edit_video_bytes = (
            edit_video.getvalue()
        )

        st.session_state.edit_video_name = (
            edit_video.name
        )

        st.session_state.ed_final = None

    suffix = Path(
        edit_video.name
    ).suffix

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as tf:

        tf.write(
            st.session_state.edit_video_bytes
        )

        edit_path = Path(
            tf.name
        )

    try:

        edit_duration = duration(
            edit_path
        )

    except Exception:

        edit_duration = 0.0

    if edit_duration <= 0:

        st.error(
            "Video duration မဖတ်နိုင်ပါ။"
        )

        st.stop()

    # --------------------------------------------------------
    # Preview timeline
    # --------------------------------------------------------

    st.subheader(
        "👀 LIVE VISUAL PREVIEW"
    )

    st.info(
        "အောက်က Preview ကိုကြည့်ပြီး "
        "Zoom / Position / Mask / "
        "စာတန်းနေရာတွေကို ချိန်ပါ။"
    )

    timeline = st.slider(
        "⏱️ Preview Timeline",
        0.0,
        float(edit_duration),
        min(
            0.0,
            float(edit_duration),
        ),
        0.1,
        key="editor_timeline",
    )

    # --------------------------------------------------------
    # Canvas
    # --------------------------------------------------------

    ratio = st.selectbox(
        "📐 Ratio / Canvas",
        [
            "Original",
            "9:16",
            "16:9",
            "1:1",
            "4:5",
        ],
        key="editor_ratio",
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        flip_h = st.checkbox(
            "↔️ Flip H",
            key="editor_flip_h",
        )

    with c2:

        flip_v = st.checkbox(
            "↕️ Flip V",
            key="editor_flip_v",
        )

    with c3:

        blur = st.checkbox(
            "🌫️ Background Blur",
            key="editor_blur",
        )

    with c4:

        mask = st.selectbox(
            "🎭 Mask",
            [
                "None",
                "Circle",
                "Rectangle",
            ],
            key="editor_mask",
        )

    # --------------------------------------------------------
    # Zoom + Position
    # --------------------------------------------------------

    st.markdown(
        "### 🔍 Zoom / Position"
    )

    z1, z2, z3 = st.columns(3)

    with z1:

        zoom = st.slider(
            "Zoom",
            0.50,
            2.50,
            1.00,
            0.05,
            key="editor_zoom",
        )

    with z2:

        px = st.slider(
            "Position X",
            -50,
            50,
            0,
            1,
            key="editor_px",
        )

    with z3:

        py = st.slider(
            "Position Y",
            -50,
            50,
            0,
            1,
            key="editor_py",
        )

    # --------------------------------------------------------
    # SRT
    # --------------------------------------------------------

    st.markdown(
        "### 📄 Subtitle"
    )

    editor_srt = st.file_uploader(
        "SRT တင်ပါ "
        "(မတင်လည်း Step 1 SRT သုံးမယ်)",
        type=["srt"],
        key="editor_srt",
    )

    source_srt = (
        st.session_state.get(
            "srt_text",
            "",
        )
    )

    if editor_srt:

        source_srt = (
            editor_srt
            .getvalue()
            .decode(
                "utf-8-sig",
                errors="replace",
            )
        )

    subtitle_rows = parse_srt(
        source_srt
    )

    current_subtitle = ""

    for row in subtitle_rows:

        if (
            row["start"]
            <= timeline
            <= row["end"]
        ):

            current_subtitle = (
                row["burmese"]
            )

            break

    # --------------------------------------------------------
    # Subtitle controls
    # --------------------------------------------------------

    s1, s2, s3 = st.columns(3)

    with s1:

        subtitle_size = st.slider(
            "🔤 စာတန်းအရွယ်",
            20,
            80,
            42,
            2,
            key="editor_sub_size",
        )

    with s2:

        subtitle_pos = st.selectbox(
            "📍 စာတန်းနေရာ",
            [
                "Top",
                "Middle",
                "Bottom",
            ],
            index=2,
            key="editor_sub_pos",
        )

    with s3:

        subtitle_outline = st.slider(
            "⭕ Outline",
            0,
            8,
            3,
            1,
            key="editor_sub_outline",
        )

    if current_subtitle:

        st.caption(
            "လက်ရှိ Dialogue:"
        )

        st.info(
            current_subtitle
        )

    else:

        st.caption(
            "ဒီအချိန်မှာ Subtitle မရှိပါ။"
        )

    # --------------------------------------------------------
    # LIVE FRAME PREVIEW
    # --------------------------------------------------------

    try:

        frame = preview_frame(
            edit_path,
            timeline,
        )

        preview = draw_preview(
            frame,
            ratio,
            zoom,
            px,
            py,
            blur,
            mask,
            current_subtitle,
            subtitle_pos,
            subtitle_size,
            subtitle_outline,
        )

        st.image(
            preview,
            caption=(
                f"🎬 LIVE PREVIEW  "
                f"{timeline:.1f}s / "
                f"{edit_duration:.1f}s"
            ),
            use_container_width=True,
        )

    except Exception as e:

        st.error(
            f"Preview မရပါ: {e}"
        )

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    st.markdown(
        "### 🔊 Audio"
    )

    voice_upload = st.file_uploader(
        "🎙️ Voiceover "
        "(.m4a / .mp3 / .wav)",
        type=[
            "m4a",
            "mp3",
            "wav",
        ],
        key="editor_voice",
    )

    voice_path = None

    if voice_upload:

        with tempfile.NamedTemporaryFile(
            suffix=Path(
                voice_upload.name
            ).suffix,
            delete=False,
        ) as vf:

            vf.write(
                voice_upload.getvalue()
            )

            voice_path = Path(
                vf.name
            )

    elif st.session_state.get(
        "voice_bytes"
    ):

        with tempfile.NamedTemporaryFile(
            suffix=".m4a",
            delete=False,
        ) as vf:

            vf.write(
                st.session_state[
                    "voice_bytes"
                ]
            )

            voice_path = Path(
                vf.name
            )

        st.success(
            "Step 2 Voiceover ကို "
            "အလိုအလျောက်သုံးမည်။"
        )

    music_upload = st.file_uploader(
        "🎵 Background Music "
        "(optional)",
        type=[
            "mp3",
            "m4a",
            "wav",
        ],
        key="editor_music",
    )

    music_path = None

    if music_upload:

        with tempfile.NamedTemporaryFile(
            suffix=Path(
                music_upload.name
            ).suffix,
            delete=False,
        ) as mf:

            mf.write(
                music_upload.getvalue()
            )

            music_path = Path(
                mf.name
            )

    a1, a2, a3 = st.columns(3)

    with a1:

        original_volume = st.slider(
            "Original Audio %",
            0,
            100,
            0,
            5,
            key="editor_original_volume",
        )

    with a2:

        voice_volume = st.slider(
            "Voiceover %",
            0,
            150,
            100,
            5,
            key="editor_voice_volume",
        )

    with a3:

        music_volume = st.slider(
            "Music %",
            0,
            80,
            15,
            5,
            key="editor_music_volume",
        )

    # --------------------------------------------------------
    # TRIM
    # --------------------------------------------------------

    st.markdown(
        "### ✂️ Trim"
    )

    t1, t2 = st.columns(2)

    with t1:

        trim_start = st.number_input(
            "Trim Start (sec)",
            min_value=0.0,
            max_value=max(
                0.0,
                edit_duration - 0.05,
            ),
            value=0.0,
            step=0.1,
            key="editor_trim_start",
        )

    with t2:

        trim_end = st.number_input(
            "Trim End (sec)",
            min_value=0.05,
            max_value=max(
                0.05,
                edit_duration,
            ),
            value=float(
                edit_duration
            ),
            step=0.1,
            key="editor_trim_end",
        )

    if (
        trim_end
        <= trim_start
    ):

        st.warning(
            "Trim End က "
            "Trim Start ထက် "
            "ကြီးရပါမယ်။"
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    quality = st.selectbox(
        "🎞️ Output Quality",
        [
            "High",
            "Good",
            "Small",
        ],
        index=1,
        key="editor_quality",
    )

    output_filename = st.text_input(
        "📁 Final Filename",
        "final_movie.mp4",
        key="editor_filename",
    )

    # --------------------------------------------------------
    # FINAL RENDER
    # --------------------------------------------------------

    if st.button(
        "🎬 FINAL RENDER",
        type="primary",
        use_container_width=True,
        key="final_render_button",
    ):

        if (
            trim_end
            <= trim_start
        ):

            st.error(
                "Trim range မမှန်ပါ။"
            )

            st.stop()

        try:

            with tempfile.TemporaryDirectory() as td:

                output = (
                    Path(td)
                    / safe_name(
                        output_filename,
                        "final_movie.mp4",
                    )
                )

                with st.spinner(
                    "🎬 Final video render "
                    "လုပ်နေသည်..."
                ):

                    render_manual(
                        edit_path,
                        voice_path,
                        source_srt,
                        float(
                            trim_start
                        ),
                        float(
                            trim_end
                        ),
                        ratio,
                        flip_h,
                        flip_v,
                        zoom,
                        px,
                        py,
                        blur,
                        mask,
                        subtitle_size,
                        subtitle_pos,
                        subtitle_outline,
                        music_path,
                        original_volume,
                        voice_volume,
                        music_volume,
                        quality,
                        output,
                    )

                st.session_state.ed_final = (
                    output.read_bytes()
                )

                st.session_state.ed_final_name = (
                    safe_name(
                        output_filename,
                        "final_movie.mp4",
                    )
                )

            st.success(
                "✅ Final Video ပြီးပါပြီ။"
            )

        except Exception as e:

            st.exception(e)


# ============================================================
# FINAL PREVIEW
# ============================================================

if st.session_state.get(
    "ed_final"
):

    st.markdown("---")

    st.subheader(
        "🎬 Final Video Preview"
    )

    st.video(
        st.session_state.ed_final
    )

    st.download_button(
        "⬇️ Download Final MP4",
        st.session_state.ed_final,
        st.session_state.get(
            "ed_final_name",
            "final_movie.mp4",
        ),
        "video/mp4",
        use_container_width=True,
        key="download_final_editor",
    )
