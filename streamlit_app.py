import os
import json
import time
import asyncio
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
from google import genai
import edge_tts


st.set_page_config(
    page_title="AI Movie Recap",
    page_icon="🎬",
    layout="wide"
)

VOICE_MAP = {
    "အမျိုးသား (သီဟ)": "my-MM-ThihaNeural",
    "အမျိုးသမီး (နီလာ)": "my-MM-NilarNeural",
}

MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

st.title("🎬 AI Movie Recap Automator")
st.caption(
    "AI dialogue detection • Natural Burmese translation • Burmese dubbing"
)


def run_cmd(cmd, timeout=3600):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-6000:] or "Command failed."
        )

    return result.stdout


def get_duration(path):
    output = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path)
        ],
        60
    )

    return float(output.strip())


def extract_json(text):
    text = (text or "").strip()

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    start = text.find("[")
    end = text.rfind("]")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def get_available_models(client):
    try:
        models = []

        for model in client.models.list():
            name = getattr(model, "name", "") or ""

            if name.startswith("models/"):
                name = name[7:]

            models.append(name)

        return models

    except Exception:
        return []


def generate_with_fallback(client, contents):
    available = get_available_models(client)

    if available:
        candidates = [
            model
            for model in MODEL_CANDIDATES
            if model in available
        ]
    else:
        candidates = MODEL_CANDIDATES

    errors = []

    for model in candidates:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents
            )

            text = getattr(response, "text", None)

            if text:
                return model, text

        except Exception as error:
            errors.append(
                f"{model}: {error}"
            )

    raise RuntimeError(
        "Gemini model အားလုံး failed:\n"
        + "\n".join(errors[-5:])
    )


def extract_audio(video_path, audio_path):
    run_cmd(
        [
            "ffmpeg",
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
            str(audio_path)
        ]
    )


def generate_tts(text, voice, output_path):
    async def create_voice():
        communicator = edge_tts.Communicate(
            text=text,
            voice=voice
        )

        await communicator.save(
            str(output_path)
        )

    asyncio.run(create_voice())


def fit_audio_to_duration(
    input_audio,
    output_audio,
    target_duration
):
    source_duration = get_duration(
        input_audio
    )

    if source_duration <= 0:
        raise RuntimeError(
            "TTS audio duration invalid."
        )

    target_duration = max(
        target_duration,
        0.05
    )

    speed = (
        source_duration /
        target_duration
    )

    speed = max(
        0.5,
        min(2.0, speed)
    )

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_audio),
            "-filter:a",
            f"atempo={speed:.6f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_audio)
        ],
        300
    )


def build_full_burmese_audio(
    segments,
    voice,
    total_duration,
    work_dir
):
    silence = (
        work_dir /
        "silence.m4a"
    )

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            str(total_duration),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(silence)
        ],
        300
    )

    audio_files = []

    for index, segment in enumerate(
        segments
    ):
        raw_tts = (
            work_dir /
            f"tts_{index}.mp3"
        )

        fitted_audio = (
            work_dir /
            f"fit_{index}.m4a"
        )

        slot_duration = (
            segment["end"] -
            segment["start"]
        )

        generate_tts(
            segment["text"],
            voice,
            raw_tts
        )

        fit_audio_to_duration(
            raw_tts,
            fitted_audio,
            slot_duration
        )

        audio_files.append(
            (
                fitted_audio,
                segment["start"]
            )
        )

    inputs = [
        "-i",
        str(silence)
    ]

    filters = [
        "[0:a]aresample=48000[base]"
    ]

    mix_inputs = [
        "[base]"
    ]

    for index, (
        audio_path,
        start_time
    ) in enumerate(
        audio_files,
        start=1
    ):
        delay_ms = max(
            0,
            int(start_time * 1000)
        )

        filters.append(
            f"[{index}:a]"
            f"aresample=48000,"
            f"adelay={delay_ms}|{delay_ms}"
            f"[a{index}]"
        )

        mix_inputs.append(
            f"[a{index}]"
        )

        inputs.extend(
            [
                "-i",
                str(audio_path)
            ]
        )

    filters.append(
        "".join(mix_inputs)
        +
        f"amix=inputs={len(mix_inputs)}:"
        f"duration=first:"
        f"dropout_transition=0,"
        f"atrim=0:{total_duration},"
        f"asetpts=PTS-STARTPTS[out]"
    )

    output_audio = (
        work_dir /
        "burmese_audio.m4a"
    )

    run_cmd(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_audio)
        ],
        3600
    )

    return output_audio


def merge_video_audio(
    video_path,
    audio_path,
    output_path
):
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path)
        ],
        3600
    )


voice_name = st.sidebar.selectbox(
    "🎙️ Voice",
    list(VOICE_MAP.keys())
)

uploaded_video = st.file_uploader(
    "🎥 Upload Video",
    type=[
        "mp4",
        "mkv",
        "mov"
    ]
)


if uploaded_video:

    st.info(
        f"📁 {uploaded_video.name} • "
        f"{uploaded_video.size / 1048576:.1f} MB"
    )

    start_button = st.button(
        "🚀 START AI DUBBING",
        type="primary",
        use_container_width=True
    )

    if start_button:

        api_key = st.secrets.get(
            "GEMINI_API_KEY",
            os.environ.get(
                "GEMINI_API_KEY"
            )
        )

        if not api_key:
            st.error(
                "GEMINI_API_KEY မတွေ့ပါ။ "
                "Streamlit Secrets ထဲထည့်ပါ။"
            )
            st.stop()

        progress = st.progress(0)
        status = st.empty()

        started_at = time.time()

        def update_progress(
            value,
            message
        ):
            value = max(
                0.0,
                min(1.0, value)
            )

            progress.progress(
                int(value * 100)
            )

            elapsed = (
                time.time() -
                started_at
            )

            if value > 0.01:
                eta = (
                    elapsed *
                    (1 - value) /
                    value
                )
            else:
                eta = 0

            status.write(
                f"**{message}**  "
                f"• {value * 100:.0f}%  "
                f"• ETA {eta:.0f}s"
            )

        try:

            with tempfile.TemporaryDirectory() as temp:

                work_dir = Path(temp)

                suffix = (
                    Path(
                        uploaded_video.name
                    ).suffix.lower()
                    or ".mp4"
                )

                video_path = (
                    work_dir /
                    f"input{suffix}"
                )

                audio_path = (
                    work_dir /
                    "movie_audio.wav"
                )

                final_audio_path = (
                    work_dir /
                    "burmese_audio.m4a"
                )

                output_path = (
                    work_dir /
                    "Myanmar_Dub.mp4"
                )

                video_path.write_bytes(
                    uploaded_video.getbuffer()
                )

                update_progress(
                    0.05,
                    "① Video ပြင်ဆင်နေသည်..."
                )

                total_duration = get_duration(
                    video_path
                )

                update_progress(
                    0.12,
                    "② Original Audio ထုတ်နေသည်..."
                )

                extract_audio(
                    video_path,
                    audio_path
                )

                update_progress(
                    0.20,
                    "③ Gemini က Dialogue ရှာနေသည်..."
                )

                client = genai.Client(
                    api_key=api_key
                )

                uploaded_audio = (
                    client.files.upload(
                        file=str(audio_path)
                    )
                )

                prompt = """
Listen to the entire uploaded audio.

Detect every meaningful spoken dialogue.

Translate each dialogue into natural spoken
Burmese suitable for a professional Myanmar
movie recap narrator.

Requirements:
- Preserve meaning.
- Preserve emotion.
- Preserve context.
- Do not invent dialogue.
- Do not skip important dialogue.
- Ignore music and sound effects.
- Keep chronological order.
- Use Burmese script.
- Times must be seconds.

Return ONLY valid JSON:

[
  {
    "start": 0.0,
    "end": 3.5,
    "text": "မြန်မာဘာသာပြန်"
  }
]

Do not use markdown.
Do not add explanations.
"""

                model_name, response = (
                    generate_with_fallback(
                        client,
                        [
                            uploaded_audio,
                            prompt
                        ]
                    )
                )

                raw_segments = extract_json(
                    response
                )

                segments = []

                for item in raw_segments:

                    try:

                        start_time = float(
                            item["start"]
                        )

                        end_time = float(
                            item["end"]
                        )

                        text = str(
                            item["text"]
                        ).strip()

                        start_time = max(
                            0.0,
                            min(
                                start_time,
                                total_duration
                            )
                        )

                        end_time = max(
                            0.0,
                            min(
                                end_time,
                                total_duration
                            )
                        )

                        if (
                            end_time >
                            start_time
                            and text
                        ):
                            segments.append(
                                {
                                    "start": start_time,
                                    "end": end_time,
                                    "text": text
                                }
                            )

                    except Exception:
                        continue

                if not segments:
                    raise RuntimeError(
                        "Gemini မှ Dialogue မရပါ။"
                    )

                update_progress(
                    0.35,
                    f"④ {len(segments)} ခု "
                    "Burmese Voice ဖန်တီးနေသည်..."
                )

                final_audio_path = (
                    build_full_burmese_audio(
                        segments,
                        VOICE_MAP[voice_name],
                        total_duration,
                        work_dir
                    )
                )

                update_progress(
                    0.85,
                    "⑤ Original Audio ဖယ်ပြီး Merge လုပ်နေသည်..."
                )

                merge_video_audio(
                    video_path,
                    final_audio_path,
                    output_path
                )

                update_progress(
                    0.97,
                    "⑥ Final Video စစ်ဆေးနေသည်..."
                )

                if (
                    not output_path.exists()
                    or
                    output_path.stat().st_size
                    < 10000
                ):
                    raise RuntimeError(
                        "Final Video မထွက်ပါ။"
                    )

                final_bytes = (
                    output_path.read_bytes()
                )

                update_progress(
                    1.0,
                    "✅ ပြီးပါပြီ!"
                )

                st.success(
                    f"🎉 Myanmar Dub Video Ready! "
                    f"• Model: {model_name}"
                )

                st.video(
                    final_bytes
                )

                st.download_button(
                    "⬇️ DOWNLOAD FINAL VIDEO",
                    data=final_bytes,
                    file_name=(
                        Path(
                            uploaded_video.name
                        ).stem
                        +
                        "_Myanmar_Dub.mp4"
                    ),
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True
                )

                with st.expander(
                    "📝 Burmese Script"
                ):

                    for index, segment in enumerate(
                        segments,
                        start=1
                    ):

                        st.write(
                            f"{index}. "
                            f"{segment['start']:.2f}s → "
                            f"{segment['end']:.2f}s"
                        )

                        st.write(
                            segment["text"]
                        )

        except Exception as error:

            st.error(
                "❌ Processing Error"
            )

            st.code(
                str(error),
                language="text"
            )
