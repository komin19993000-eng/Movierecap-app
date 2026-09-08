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

st.set_page_config(
    page_title="Movie Dubbing AI",
    page_icon="🎬",
    layout="wide"
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
    "resource exhausted"
)


def run_cmd(args, timeout=1800):
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout
    )


def duration(path):
    r = run_cmd(
        [FFMPEG, "-hide_banner", "-i", str(path)],
        120
    )

    m = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        r.stderr or ""
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
            str(out)
        ],
        900
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


@st.cache_resource
def client():
    key = st.secrets.get(
        "GEMINI_API_KEY",
        os.getenv("GEMINI_API_KEY", "")
    )

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ထဲမှာ ထည့်ပါ။"
        )

    return genai.Client(api_key=key)


def temporary_error(e):
    s = str(e).lower()
    return any(
        x.lower() in s
        for x in TEMP_WORDS
    )


def model_list(c):
    try:
        names = []

        for m in c.models.list():
            n = getattr(m, "name", "")

            if n:
                names.append(
                    n.replace("models/", "")
                )

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
        flags=re.I
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
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

    for x in items:

        if not isinstance(x, dict):
            continue

        try:
            s = float(x.get("start", 0))
            e = float(x.get("end", 0))
        except Exception:
            continue

        t = str(
            x.get("burmese", "")
        ).strip()

        s = max(0, min(s, dur))
        e = max(0, min(e, dur))

        if t and e - s >= 0.20:
            out.append(
                {
                    "start": s,
                    "end": e,
                    "burmese": t
                }
            )

    out.sort(
        key=lambda x: x["start"]
    )

    cleaned = []

    for x in out:

        if (
            cleaned
            and abs(
                x["start"]
                - cleaned[-1]["start"]
            ) < 0.05
            and x["burmese"]
            == cleaned[-1]["burmese"]
        ):
            cleaned[-1]["end"] = max(
                cleaned[-1]["end"],
                x["end"]
            )
        else:
            cleaned.append(x)

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
                    getattr(cur, "state", "")
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
        c.files.upload(
            file=str(audio)
        )
    )

    prompt = f"""
You are a professional movie dubbing editor.

Analyze the uploaded movie audio and find ALL meaningful spoken dialogue.

Do not invent dialogue.

Keep chronological order.

Return approximate start/end timestamps in seconds.

Translate every line into natural conversational Burmese suitable for professional movie dubbing.

Preserve:
- meaning
- emotion
- names
- relationships
- context

Exclude:
- music
- sound effects
- background noise

Keep each Burmese line concise enough to fit its timestamp.

Do not merge unrelated lines.

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
                                    None
                                )
                                or "audio/wav"
                            )
                        ),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.15
                    )
                )

                seg = normalize(
                    json.loads(
                        clean_json(
                            getattr(
                                resp,
                                "text",
                                ""
                            )
                        )
                    ),
                    dur
                )

                if seg:
                    return seg, model

                raise RuntimeError(
                    "Dialogue မတွေ့ပါ။"
                )

            except Exception as e:

                errors.append(
                    f"{model} attempt "
                    f"{attempt + 1}: {e}"
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


async def tts_async(text, voice, out):

    await edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
        volume="+0%"
    ).save(str(out))


def tts(text, voice, out):

    asyncio.run(
        tts_async(
            text,
            voice,
            out
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
        min(float(speed), 3.0)
    )

    f = []

    while speed > 2:
        f.append("atempo=2")
        speed /= 2

    while speed < 0.5:
        f.append("atempo=.5")
        speed /= 0.5

    f.append(
        f"atempo={speed:.6f}"
    )

    return ",".join(f)


def fit_tts(src, out, slot):

    d = duration(src)

    speed = max(
        0.5,
        min(
            d / max(slot, 0.25),
            3.0
        )
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
            str(out)
        ],
        180
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
            "-"
        ],
        300
    )

    return r.returncode == 0


def make_burmese_audio(
    segments,
    voice,
    dur,
    work,
    progress
):

    files = []
    total = len(segments)

    for i, s in enumerate(
        segments,
        1
    ):

        progress(
            (i - 1)
            / max(total, 1),
            f"TTS {i}/{total}"
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
            s["burmese"],
            voice,
            raw
        )

        fit_tts(
            raw,
            fitted,
            max(
                0.25,
                s["end"] - s["start"]
            )
        )

        files.append(
            (
                s["start"],
                fitted
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
        "error"
    ]

    for _, f in files:
        cmd += [
            "-i",
            str(f)
        ]

    filters = []
    labels = []

    for i, (start, _) in enumerate(files):

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
        str(out)
    ]

    r = run_cmd(
        cmd,
        1800
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
        "Burmese audio OK"
    )

    return out


def validate_video(path):

    if (
        not path.exists()
        or path.stat().st_size < 10000
    ):
        return (
            False,
            "Final video file မမှန်ပါ။"
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
            "-"
        ],
        300
    )

    if v.returncode:
        return (
            False,
            "Video decode မအောင်မြင်ပါ။"
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
            "-"
        ],
        300
    )

    if a.returncode:
        return (
            False,
            "Final video ထဲမှာ "
            "Audio track မရှိပါ "
            "သို့မဟုတ် decode မရပါ။"
        )

    return True, "OK"


def export_video(
    video,
    audio,
    out
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

            str(out)
        ],
        1800
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

                str(out)
            ],
            3600
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
            "မအောင်မြင်ပါ။\n"
            + msg
        )


st.title("🎬 Movie Dubbing AI")

st.caption(
    "Video → AI dialogue → "
    "Burmese dubbing → Preview → Download"
)

if "final_bytes" not in st.session_state:
    st.session_state.final_bytes = None

if "final_name" not in st.session_state:
    st.session_state.final_name = (
        "dubbed_video.mp4"
    )


uploaded = st.file_uploader(
    "🎥 Video တင်ပါ",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm"
    ]
)

voice_name = st.selectbox(
    "🎙️ အသံ",
    list(VOICES)
)

start = st.button(
    "🚀 Start Dubbing",
    type="primary",
    use_container_width=True
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
                work
                / "input_video"
            )

            video.write_bytes(
                uploaded.getbuffer()
            )

            dur = duration(video)

            status = st.empty()

            bar = st.progress(0.0)

            def set_status(text):
                status.info(text)

            set_status(
                "1/5 Video စစ်နေသည်..."
            )

            audio = (
                work
                / "original_audio.wav"
            )

            extract_audio(
                video,
                audio
            )

            bar.progress(0.10)

            set_status(
                "2/5 AI dialogue "
                "နားထောင်/ဘာသာပြန်နေသည်..."
            )

            seg, model = analyze(
                client(),
                audio,
                dur,
                set_status
            )

            bar.progress(0.35)

            st.success(
                f"AI model: {model} • "
                f"Dialogue: {len(seg)} lines"
            )

            set_status(
                "3/5 Burmese voice "
                "ထုတ်နေသည်..."
            )

            def prog(p, t):

                bar.progress(
                    0.35 + 0.45 * p
                )

                status.info(t)

            burmese = make_burmese_audio(
                seg,
                VOICES[voice_name],
                dur,
                work,
                prog
            )

            set_status(
                "4/5 Final video "
                "ပြုလုပ်နေသည်..."
            )

            final = (
                work
                / "final_dubbed.mp4"
            )

            export_video(
                video,
                burmese,
                final
            )

            bar.progress(0.95)

            data = final.read_bytes()

            st.session_state.final_bytes = data

            st.session_state.final_name = (
                "dubbed_video.mp4"
            )

            bar.progress(1.0)

            status.success(
                "5/5 ပြီးပါပြီ — "
                "Video + Burmese audio "
                "validation OK"
            )

    except Exception as e:

        st.exception(e)


if st.session_state.final_bytes:

    st.subheader(
        "🎬 Preview"
    )

    st.video(
        st.session_state.final_bytes
    )

    st.download_button(
        "⬇️ Download Final Video",
        st.session_state.final_bytes,
        st.session_state.final_name,
        "video/mp4",
        use_container_width=True
    )
