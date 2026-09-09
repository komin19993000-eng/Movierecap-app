import os
import re
import json
import time
import random
import subprocess
import tempfile
from pathlib import Path
import asyncio

import requests
import streamlit as st
import edge_tts
import imageio_ffmpeg
from google import genai


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Movie Translator AI",
    page_icon="🎬",
    layout="wide"
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

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


# =========================================================
# HELPERS
# =========================================================

def get_secret(name):
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""

    if value:
        return str(value).strip()

    return os.getenv(name, "").strip()


def gemini_client():
    key = get_secret("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ထဲမှာ ထည့်ထားတာ စစ်ပါ။"
        )

    return genai.Client(api_key=key)


def deepgram_key():
    key = get_secret("DEEPGRAM_API_KEY")

    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY မတွေ့ပါ။ "
            "Streamlit Secrets ထဲမှာ ထည့်ထားတာ စစ်ပါ။"
        )

    return key


def run_cmd(command, timeout=900):
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout
    )


# =========================================================
# FFMPEG
# =========================================================

def extract_audio(video_path, audio_path):
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
        900
    )

    if (
        result.returncode != 0
        or not audio_path.exists()
        or audio_path.stat().st_size < 1000
    ):
        raise RuntimeError(
            "Video ထဲက audio ထုတ်မရပါ။\n\n"
            + (result.stderr or "")
        )


# =========================================================
# DEEPGRAM
# =========================================================

def words_to_utterances(words):
    """
    Deepgram utterances မရခဲ့ရင် word timestamps ကနေ
    sentence/utterance ပြန်တည်ဆောက်ပေးသည်။
    """

    result = []
    current_words = []

    for word in words:
        current_words.append(word)

        punct = str(word.get("punctuated_word", word.get("word", "")))

        if (
            punct.endswith(".")
            or punct.endswith("?")
            or punct.endswith("!")
            or punct.endswith("。")
            or punct.endswith("？")
            or punct.endswith("！")
        ):
            start = current_words[0].get("start", 0)
            end = current_words[-1].get("end", start)

            text = " ".join(
                str(x.get("punctuated_word", x.get("word", "")))
                for x in current_words
            ).strip()

            if text:
                result.append(
                    {
                        "start": float(start),
                        "end": float(end),
                        "transcript": text,
                    }
                )

            current_words = []

    if current_words:
        start = current_words[0].get("start", 0)
        end = current_words[-1].get("end", start)

        text = " ".join(
            str(x.get("punctuated_word", x.get("word", "")))
            for x in current_words
        ).strip()

        if text:
            result.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "transcript": text,
                }
            )

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
    }

    headers = {
        "Authorization": f"Token {deepgram_key()}",
        "Content-Type": "audio/wav",
    }

    data = audio_path.read_bytes()

    last_error = ""

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

                results = obj.get("results", {})

                utterances = results.get(
                    "utterances",
                    []
                ) or []

                if utterances:

                    cleaned = []

                    for item in utterances:

                        text = str(
                            item.get("transcript", "")
                        ).strip()

                        start = item.get("start")
                        end = item.get("end")

                        if (
                            text
                            and start is not None
                            and end is not None
                        ):
                            cleaned.append(
                                {
                                    "start": float(start),
                                    "end": float(end),
                                    "transcript": text,
                                }
                            )

                    if cleaned:
                        return cleaned

                channels = results.get(
                    "channels",
                    []
                )

                if channels:

                    alternatives = channels[0].get(
                        "alternatives",
                        []
                    )

                    if alternatives:

                        words = alternatives[0].get(
                            "words",
                            []
                        ) or []

                        if words:
                            return words_to_utterances(words)

                raise RuntimeError(
                    "Deepgram က transcript မပြန်ပေးပါ။"
                )

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:700]}"
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

            last_error = str(e)

        time.sleep(
            (2 ** attempt)
            + random.random()
        )

    raise RuntimeError(
        "Deepgram STT မအောင်မြင်ပါ။\n\n"
        + last_error
    )


# =========================================================
# GEMINI TRANSLATION
# =========================================================

def translate_batch(client, items):

    payload = []

    for index, item in enumerate(items, 1):

        payload.append(
            {
                "id": index,
                "text": item["transcript"],
            }
        )

    prompt = f"""
You are a professional Burmese movie subtitle translator.

Translate the following movie dialogue into natural,
conversational Myanmar Burmese.

IMPORTANT RULES:

1. Translate meaning naturally, not word-for-word.
2. Preserve names and important terms.
3. Keep the emotional tone.
4. Do not add explanations.
5. Do not summarize.
6. Do not remove dialogue.
7. Keep each translation reasonably short because
   it must fit the original dialogue timing.
8. Output ONLY valid JSON.
9. Use exactly the same IDs.
10. The output must contain Burmese translation only.

INPUT:

{json.dumps(payload, ensure_ascii=False)}

OUTPUT FORMAT:

[
  {{
    "id": 1,
    "burmese": "မြန်မာဘာသာပြန်"
  }}
]
"""

    last_error = ""

    for model in GEMINI_MODELS:

        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                    },
                )

                text = str(
                    getattr(response, "text", "")
                    or ""
                ).strip()

                if not text:
                    raise RuntimeError(
                        "Gemini response အလွတ်ဖြစ်နေပါသည်။"
                    )

                text = re.sub(
                    r"^```json\s*",
                    "",
                    text,
                    flags=re.IGNORECASE
                )

                text = re.sub(
                    r"^```\s*",
                    "",
                    text
                )

                text = re.sub(
                    r"\s*```$",
                    "",
                    text
                )

                data = json.loads(text)

                if not isinstance(data, list):
                    raise RuntimeError(
                        "Gemini JSON format မမှန်ပါ။"
                    )

                translated = {}

                for row in data:

                    row_id = row.get("id")
                    burmese = str(
                        row.get("burmese", "")
                    ).strip()

                    if row_id is not None and burmese:
                        translated[int(row_id)] = burmese

                if len(translated) != len(items):
                    raise RuntimeError(
                        "Gemini က subtitle အချို့ကို "
                        "မပြန်ပေးပါ။"
                    )

                return [
                    translated[i]
                    for i in range(1, len(items) + 1)
                ]

            except Exception as e:

                last_error = (
                    f"{model} attempt "
                    f"{attempt + 1}: {e}"
                )

                time.sleep(
                    2 + attempt * 2
                )

    raise RuntimeError(
        "Gemini Translation မအောင်မြင်ပါ။\n\n"
        + last_error
    )


def translate_segments(utterances, progress_callback=None):

    client = gemini_client()

    all_translations = []

    batch_size = 12

    total = len(utterances)

    for start in range(
        0,
        total,
        batch_size
    ):

        batch = utterances[
            start:start + batch_size
        ]

        translated = translate_batch(
            client,
            batch
        )

        all_translations.extend(
            translated
        )

        if progress_callback:
            progress_callback(
                min(
                    1.0,
                    len(all_translations) / total
                )
            )

    segments = []

    for item, burmese in zip(
        utterances,
        all_translations
    ):

        start = float(item["start"])
        end = float(item["end"])

        if end <= start:
            continue

        segments.append(
            {
                "start": start,
                "end": end,
                "original": item["transcript"],
                "burmese": burmese,
            }
        )

    return segments


# =========================================================
# SRT
# =========================================================

def srt_time(seconds):

    seconds = max(
        0.0,
        float(seconds)
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(
        seconds % 60
    )

    milliseconds = int(
        round(
            (seconds - int(seconds))
            * 1000
        )
    )

    if milliseconds >= 1000:
        milliseconds = 0
        secs += 1

    if secs >= 60:
        secs = 0
        minutes += 1

    if minutes >= 60:
        minutes = 0
        hours += 1

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def make_srt(segments):

    blocks = []

    for index, item in enumerate(
        segments,
        1
    ):

        blocks.append(
            f"{index}\n"
            f"{srt_time(item['start'])} --> "
            f"{srt_time(item['end'])}\n"
            f"{item['burmese'].strip()}\n"
        )

    return "\n".join(blocks)


def parse_srt(text):

    text = text.replace(
        "\ufeff",
        ""
    )

    text = text.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    blocks = re.split(
        r"\n\s*\n",
        text.strip()
    )

    segments = []

    time_pattern = re.compile(
        r"(\d{1,2}):"
        r"(\d{2}):"
        r"(\d{2})[,.](\d{1,3})"
        r"\s*-->\s*"
        r"(\d{1,2}):"
        r"(\d{2}):"
        r"(\d{2})[,.](\d{1,3})"
    )

    def parse_time(h, m, s, ms):

        ms = ms.ljust(
            3,
            "0"
        )

        return (
            int(h) * 3600
            + int(m) * 60
            + int(s)
            + int(ms[:3]) / 1000
        )

    for block in blocks:

        lines = [
            x.strip()
            for x in block.split("\n")
            if x.strip()
        ]

        if len(lines) < 3:
            continue

        time_line = None
        time_index = -1

        for i, line in enumerate(lines):

            if "-->" in line:
                time_line = line
                time_index = i
                break

        if not time_line:
            continue

        match = time_pattern.search(
            time_line
        )

        if not match:
            continue

        start = parse_time(
            match.group(1),
            match.group(2),
            match.group(3),
            match.group(4)
        )

        end = parse_time(
            match.group(5),
            match.group(6),
            match.group(7),
            match.group(8)
        )

        subtitle_text = "\n".join(
            lines[time_index + 1:]
        ).strip()

        if not subtitle_text:
            continue

        if end <= start:
            continue

        segments.append(
            {
                "start": start,
                "end": end,
                "burmese": subtitle_text,
            }
        )

    segments.sort(
        key=lambda x: x["start"]
    )

    # Automatically fix overlaps.
    for i in range(
        1,
        len(segments)
    ):

        previous = segments[i - 1]
        current = segments[i]

        if current["start"] < previous["end"]:

            current["start"] = previous["end"]

            if current["end"] <= current["start"]:
                current["end"] = (
                    current["start"] + 0.1
                )

    return segments


# =========================================================
# EDGE TTS
# =========================================================

async def edge_tts_save(
    text,
    voice,
    rate,
    pitch,
    output_path
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


def create_tts(
    text,
    voice,
    rate,
    pitch,
    output_path
):

    text = str(text).strip()

    if not text:
        raise RuntimeError(
            "TTS text အလွတ်ဖြစ်နေပါသည်။"
        )

    last_error = ""

    for attempt in range(3):

        try:

            if output_path.exists():
                output_path.unlink()

            asyncio.run(
                edge_tts_save(
                    text,
                    voice,
                    rate,
                    pitch,
                    output_path
                )
            )

            if (
                output_path.exists()
                and output_path.stat().st_size >= 1000
            ):
                return

            raise RuntimeError(
                "TTS audio file အလွတ်ဖြစ်နေပါသည်။"
            )

        except Exception as e:

            last_error = str(e)

            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass

            time.sleep(
                2 + attempt * 2
            )

    raise RuntimeError(
        "Edge TTS မအောင်မြင်ပါ။\n\n"
        + last_error
    )


# =========================================================
# AUDIO TIMING
# =========================================================

def atempo_chain(factor):

    factor = float(factor)

    factor = max(
        0.25,
        min(
            4.0,
            factor
        )
    )

    parts = []

    while factor < 0.5:

        parts.append(
            "atempo=0.5"
        )

        factor /= 0.5

    while factor > 2.0:

        parts.append(
            "atempo=2.0"
        )

        factor /= 2.0

    parts.append(
        f"atempo={factor:.6f}"
    )

    return ",".join(parts)


def audio_duration(audio_path):

    result = run_cmd(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(audio_path),
        ],
        60
    )

    text = (
        result.stderr
        or result.stdout
        or ""
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        text
    )

    if not match:
        raise RuntimeError(
            "Audio duration မဖတ်နိုင်ပါ။"
        )

    hours = float(
        match.group(1)
    )

    minutes = float(
        match.group(2)
    )

    seconds = float(
        match.group(3)
    )

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )


def fit_clip(
    source,
    output,
    slot,
    user_speed
):

    raw_duration = audio_duration(
        source
    )

    slot = max(
        0.05,
        float(slot)
    )

    user_speed = float(
        user_speed
    )

    # Natural timing correction + user's speed.
    final_speed = (
        raw_duration / slot
    ) * user_speed

    final_speed = max(
        0.25,
        min(
            4.0,
            final_speed
        )
    )

    filter_chain = (
        atempo_chain(final_speed)
        + ",apad,"
        + f"atrim=duration={slot:.3f}"
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
            filter_chain,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(output),
        ],
        300
    )

    if (
        result.returncode != 0
        or not output.exists()
        or output.stat().st_size < 1000
    ):
        raise RuntimeError(
            "Voice timing ပြင်မရပါ။\n\n"
            + (result.stderr or "")
        )


# =========================================================
# BUILD FINAL VOICEOVER
# =========================================================

def build_voiceover(
    segments,
    voice,
    style,
    speed,
    progress_callback=None
):

    style_config = VOICE_STYLES[
        style
    ]

    voice_name = VOICES[
        voice
    ]

    with tempfile.TemporaryDirectory() as temp_dir:

        temp = Path(temp_dir)

        clips = []

        total = len(segments)

        if total == 0:
            raise RuntimeError(
                "SRT ထဲမှာ subtitle မတွေ့ပါ။"
            )

        for index, segment in enumerate(
            segments
        ):

            text = segment[
                "burmese"
            ].strip()

            if not text:
                continue

            slot = (
                float(segment["end"])
                - float(segment["start"])
            )

            if slot <= 0:
                continue

            raw = temp / (
                f"raw_{index:05d}.mp3"
            )

            fitted = temp / (
                f"fit_{index:05d}.m4a"
            )

            create_tts(
                text,
                voice_name,
                style_config["rate"],
                style_config["pitch"],
                raw
            )

            fit_clip(
                raw,
                fitted,
                slot,
                speed
            )

            clips.append(
                (
                    fitted,
                    float(segment["start"])
                )
            )

            if progress_callback:
                progress_callback(
                    min(
                        0.85,
                        (
                            index + 1
                        ) / total * 0.85
                    )
                )

        if not clips:
            raise RuntimeError(
                "Voiceover ထုတ်ဖို့ subtitle မရှိပါ။"
            )

        output = temp / (
            "burmese_voiceover.m4a"
        )

        input_args = []

        filter_parts = []

        for i, (clip, start) in enumerate(
            clips
        ):

            input_args.extend(
                [
                    "-i",
                    str(clip)
                ]
            )

            delay = max(
                0,
                int(
                    round(
                        start * 1000
                    )
                )
            )

            filter_parts.append(
                f"[{i}:a]"
                f"adelay={delay}|{delay}"
                f"[a{i}]"
            )

        mix_inputs = "".join(
            f"[a{i}]"
            for i in range(
                len(clips)
            )
        )

        filter_parts.append(
            f"{mix_inputs}"
            f"amix=inputs={len(clips)}:"
            f"duration=longest:"
            f"dropout_transition=0,"
            f"aresample=48000"
            f"[out]"
        )

        filter_complex = ";".join(
            filter_parts
        )

        result = run_cmd(
            [
                FFMPEG,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                *input_args,
                "-filter_complex",
                filter_complex,
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
            ],
            900
        )

        if (
            result.returncode != 0
            or not output.exists()
            or output.stat().st_size < 1000
        ):
            raise RuntimeError(
                "Final voiceover မထုတ်နိုင်ပါ။\n\n"
                + (result.stderr or "")
            )

        final_bytes = output.read_bytes()

        if progress_callback:
            progress_callback(1.0)

        return final_bytes


# =========================================================
# SESSION STATE
# =========================================================

if "srt_text" not in st.session_state:
    st.session_state["srt_text"] = ""

if "voice_bytes" not in st.session_state:
    st.session_state["voice_bytes"] = None


# =========================================================
# UI
# =========================================================

st.title("🎬 Movie Translator AI")

st.caption(
    "Video → Myanmar SRT → Myanmar Voiceover"
)

st.divider()


# =========================================================
# STEP 1
# =========================================================

st.header("1️⃣ Video → မြန်မာ SRT")

video_file = st.file_uploader(
    "Video Upload",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm"
    ],
    key="video_upload"
)

if video_file:

    st.info(
        "Video ကို Deepgram နဲ့ transcript ထုတ်ပြီး "
        "Gemini နဲ့ သဘာဝကျတဲ့ မြန်မာဘာသာပြန် SRT "
        "ထုတ်ပေးပါမယ်။"
    )

    if st.button(
        "📝 မြန်မာ SRT ထုတ်မယ်",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(0)

        status = st.empty()

        try:

            with tempfile.TemporaryDirectory() as temp_dir:

                temp = Path(temp_dir)

                video_path = temp / (
                    video_file.name
                )

                audio_path = temp / (
                    "audio.wav"
                )

                video_path.write_bytes(
                    video_file.getvalue()
                )

                status.info(
                    "🎧 Step 1/3 — Video audio ထုတ်နေပါတယ်..."
                )

                extract_audio(
                    video_path,
                    audio_path
                )

                progress.progress(15)

                status.info(
                    "🗣️ Step 2/3 — Deepgram transcript လုပ်နေပါတယ်..."
                )

                utterances = deepgram_transcribe(
                    audio_path
                )

                if not utterances:
                    raise RuntimeError(
                        "Deepgram transcript မရပါ။"
                    )

                progress.progress(40)

                status.info(
                    f"🤖 Step 3/3 — Gemini ဘာသာပြန်နေပါတယ်... "
                    f"({len(utterances)} lines)"
                )

                def update_translation(value):
                    progress.progress(
                        int(
                            40 + value * 50
                        )
                    )

                segments = translate_segments(
                    utterances,
                    update_translation
                )

                srt_text = make_srt(
                    segments
                )

                st.session_state[
                    "srt_text"
                ] = srt_text

                progress.progress(100)

                status.success(
                    f"✅ SRT အောင်မြင်ပါပြီ — "
                    f"{len(segments)} subtitles"
                )

        except Exception as e:

            progress.empty()

            status.error(
                "❌ SRT ထုတ်ရာမှာ Error ဖြစ်ပါတယ်။"
            )

            st.exception(e)


if st.session_state[
    "srt_text"
]:

    st.subheader(
        "📄 Myanmar SRT Preview"
    )

    st.text_area(
        "SRT",
        value=st.session_state[
            "srt_text"
        ],
        height=350,
        key="srt_preview"
    )

    st.download_button(
        "⬇️ Myanmar SRT Download",
        data=st.session_state[
            "srt_text"
        ].encode("utf-8"),
        file_name="myanmar_translation.srt",
        mime="application/x-subrip",
        use_container_width=True
    )


st.divider()


# =========================================================
# STEP 2
# =========================================================

st.header("2️⃣ SRT → မြန်မာ Voiceover")

srt_file = st.file_uploader(
    "SRT Upload (သို့) Step 1 က SRT ကိုသုံးပါ",
    type=["srt"],
    key="srt_upload"
)

selected_srt = ""

if srt_file:

    selected_srt = srt_file.getvalue().decode(
        "utf-8-sig",
        errors="replace"
    )

elif st.session_state[
    "srt_text"
]:

    selected_srt = st.session_state[
        "srt_text"
    ]

if selected_srt:

    st.success(
        "✅ SRT ရပါပြီ"
    )

    voice = st.selectbox(
        "🎙️ Voice",
        list(VOICES.keys())
    )

    style = st.selectbox(
        "🎭 Voice Style",
        list(VOICE_STYLES.keys())
    )

    speed = st.slider(
        "⚡ Speed",
        min_value=0.70,
        max_value=1.30,
        value=1.00,
        step=0.05
    )

    if st.button(
        "🗣️ Voiceover ထုတ်မယ်",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(0)

        status = st.empty()

        try:

            segments = parse_srt(
                selected_srt
            )

            if not segments:
                raise RuntimeError(
                    "SRT format မမှန်ပါ "
                    "သို့မဟုတ် subtitle မတွေ့ပါ။"
                )

            status.info(
                f"🔍 SRT timestamp စစ်ပြီးပါပြီ — "
                f"{len(segments)} lines"
            )

            progress.progress(5)

            def update_voice_progress(value):
                progress.progress(
                    int(
                        5 + value * 95
                    )
                )

            voice_bytes = build_voiceover(
                segments=segments,
                voice=voice,
                style=style,
                speed=speed,
                progress_callback=update_voice_progress
            )

            st.session_state[
                "voice_bytes"
            ] = voice_bytes

            progress.progress(100)

            status.success(
                "✅ Myanmar Voiceover အောင်မြင်ပါပြီ!"
            )

        except Exception as e:

            progress.empty()

            status.error(
                "❌ Voiceover ထုတ်ရာမှာ Error ဖြစ်ပါတယ်။"
            )

            st.exception(e)


if st.session_state[
    "voice_bytes"
]:

    st.subheader(
        "🔊 Voiceover Preview"
    )

    st.audio(
        st.session_state[
            "voice_bytes"
        ],
        format="audio/mp4"
    )

    st.download_button(
        "⬇️ Voiceover M4A Download",
        data=st.session_state[
            "voice_bytes"
        ],
        file_name="burmese_voiceover.m4a",
        mime="audio/mp4",
        use_container_width=True
    )


st.divider()


# =========================================================
# STEP 3
# =========================================================

st.header("3️⃣ ပြီးပါပြီ ✅")

st.write(
    "ဒီ App က Video ထဲကို Voiceover ပြန်ထည့်ခြင်း၊ "
    "Original Audio ဖျက်ခြင်း၊ Video Editing ပြုလုပ်ခြင်း "
    "မလုပ်ပါ။"
)

st.write(
    "ထုတ်ထားတဲ့ SRT နဲ့ M4A ကို Download လုပ်ပြီး "
    "နောက်ပိုင်းမှာ ကိုယ်တိုင် Video Editing လုပ်နိုင်ပါတယ်။"
)
