import os
import re
import json
import time
import shutil
import asyncio
import subprocess
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
import edge_tts


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Myanmar AI Dubbing Studio",
    page_icon="🎬",
    layout="wide"
)

APP_TITLE = "🎬 Myanmar AI Dubbing Studio"

GEMINI_MODEL = "gemini-3.7-flash"

VOICE_THIHA = "my-MM-ThihaNeural"
VOICE_NILAR = "my-MM-NilarNeural"

MAX_TRANSLATION_BATCH = 12

WORK_ROOT = Path("temp_workspace")


# ============================================================
# UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0,242,254,0.08), transparent 30%),
        radial-gradient(circle at 90% 90%, rgba(79,172,254,0.08), transparent 30%),
        #0b0e12;
    color: #e8edf3;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        rgba(0,242,254,0.12),
        rgba(79,172,254,0.05)
    );
    border: 1px solid rgba(0,242,254,0.18);
    margin-bottom: 25px;
}

.hero h1 {
    margin: 0;
    font-size: 38px;
}

.hero p {
    color: #9ca8b5;
    margin-top: 8px;
}

.stage-card {
    padding: 12px 16px;
    border-radius: 12px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 8px;
}

.success-box {
    padding: 18px;
    border-radius: 14px;
    background: rgba(0, 255, 170, 0.08);
    border: 1px solid rgba(0, 255, 170, 0.25);
}

.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    border: none;
    font-weight: 800;
    font-size: 16px;
    background: linear-gradient(90deg, #00f2fe, #4facfe);
    color: #061018;
}

div[data-testid="stFileUploader"] {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>🎬 Myanmar AI Dubbing Studio</h1>
    <p>
        AI Transcript → Natural Burmese → Myanmar Voiceover →
        Dialogue Sync → Final Video
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "completed": False,
    "video_bytes": None,
    "script": "",
    "segments": [],
    "last_error": "",
    "processing": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def run_cmd(cmd, timeout=None, quiet=False):
    """
    Run subprocess safely.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            error = result.stderr[-4000:] if result.stderr else "Unknown FFmpeg error."
            raise RuntimeError(error)

        return result

    except subprocess.TimeoutExpired:
        raise RuntimeError("Process timeout ဖြစ်သွားပါတယ်။")


def check_binary(name):
    return shutil.which(name) is not None


def get_duration(path):
    """
    Get media duration with ffprobe.
    """
    try:
        result = run_cmd(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path)
            ],
            timeout=30
        )

        value = result.stdout.strip()

        if not value:
            return 0.0

        return float(value)

    except Exception:
        return 0.0


def clean_json_text(text):
    """
    Gemini sometimes returns ```json ... ```
    Remove markdown fences.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I)
        text = re.sub(r"```$", "", text.strip())

    return text.strip()


def safe_json_loads(text):
    text = clean_json_text(text)

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        # Try extracting first JSON array.
        start = text.find("[")
        end = text.rfind("]")

        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])

        raise


def format_time(seconds):
    seconds = max(0, int(seconds))

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"

    return f"{m:02d}:{s:02d}"


def set_status(stage, message, progress, started_at):
    elapsed = time.time() - started_at

    st.session_state.status_box.markdown(
        f"""
        <div class="stage-card">
            <b>{stage}</b><br>
            {message}
            <br>
            <small>
                Progress: {progress:.0f}% |
                Elapsed: {format_time(elapsed)}
            </small>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.session_state.progress_bar.progress(
        min(max(progress / 100.0, 0.0), 1.0)
    )


# ============================================================
# TTS
# ============================================================

async def generate_tts_async(text, output_path, voice):
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice
    )

    await communicate.save(str(output_path))


def generate_tts(text, output_path, voice):
    asyncio.run(
        generate_tts_async(
            text,
            output_path,
            voice
        )
    )


# ============================================================
# ATEMPO FILTER
# ============================================================

def build_atempo_chain(speed):
    """
    FFmpeg atempo accepts 0.5 - 2.0 per filter.
    Build a safe chain for arbitrary speed.
    """

    speed = max(0.25, min(float(speed), 4.0))

    filters = []

    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5

    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0

    filters.append(f"atempo={speed:.6f}")

    return ",".join(filters)


# ============================================================
# FIT AUDIO TO EXACT DURATION
# ============================================================

def fit_audio_to_duration(input_audio, output_audio, target_duration):
    """
    Adjust TTS audio to target dialogue duration.

    Important:
    We DO NOT change video speed.

    The original video timeline remains the master timeline.
    """

    source_duration = get_duration(input_audio)

    if source_duration <= 0:
        raise RuntimeError("TTS audio duration မရပါ။")

    target_duration = max(float(target_duration), 0.10)

    speed = source_duration / target_duration

    # Keep voice reasonably natural.
    # If extremely different, cap speed and trim/pad afterward.
    speed = max(0.65, min(speed, 1.75))

    atempo = build_atempo_chain(speed)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_audio),

        "-filter:a",
        f"{atempo},"
        f"apad,"
        f"atrim=duration={target_duration:.3f}",

        "-ar", "24000",
        "-ac", "1",
        "-c:a", "aac",
        "-b:a", "128k",

        str(output_audio)
    ]

    run_cmd(cmd, timeout=120)

    return get_duration(output_audio)


# ============================================================
# CREATE SILENCE
# ============================================================

def create_silence(path, duration):
    duration = max(0.0, float(duration))

    if duration <= 0.01:
        return

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=24000:cl=mono",
            "-t", f"{duration:.3f}",
            "-c:a", "aac",
            "-b:a", "128k",
            str(path)
        ],
        timeout=60
    )


# ============================================================
# TRANSCRIPT
# ============================================================

def generate_transcript(client, audio_path):
    uploaded = client.files.upload(
        file=str(audio_path)
    )

    prompt = """
You are a professional movie transcription engine.

Listen carefully to the uploaded movie audio.

Create a timestamped transcript.

Rules:
1. Detect every meaningful spoken dialogue.
2. Do NOT summarize.
3. Do NOT translate.
4. Preserve the actual spoken meaning.
5. Split dialogue naturally.
6. Each segment must have:
   - start: number of seconds
   - end: number of seconds
   - text: original spoken dialogue
7. Ignore music and normal background noise.
8. Do not invent dialogue.
9. Keep chronological order.
10. Use decimal seconds.

Return ONLY a JSON array:

[
  {
    "start": 0.0,
    "end": 2.5,
    "text": "..."
  }
]
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[uploaded, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    data = safe_json_loads(response.text)

    if not isinstance(data, list):
        raise RuntimeError("Transcript JSON format မမှန်ပါ။")

    cleaned = []

    for item in data:
        try:
            start = float(item.get("start", 0))
            end = float(item.get("end", 0))
            text = str(item.get("text", "")).strip()

            if not text:
                continue

            if end <= start:
                continue

            cleaned.append({
                "start": start,
                "end": end,
                "original": text,
                "text": ""
            })

        except Exception:
            continue

    if not cleaned:
        raise RuntimeError("Dialogue transcript မရပါ။")

    return cleaned


# ============================================================
# BURMESE TRANSLATION
# ============================================================

def translate_batch(client, batch):
    payload = []

    for i, seg in enumerate(batch):
        payload.append({
            "id": i,
            "text": seg["original"]
        })

    prompt = f"""
You are a professional Burmese movie-recap script writer.

Translate/adapt the following movie dialogue into NATURAL SPOKEN BURMESE.

Goal:
- Sounds like a Burmese movie recap narrator.
- Natural spoken Burmese.
- Easy to listen to.
- Preserve the meaning.
- Do not translate word-for-word when that sounds unnatural.
- Keep character names consistent.
- Do not add information that isn't present.
- Do not remove important meaning.
- Keep each dialogue as one corresponding item.

IMPORTANT:
Return ONLY JSON.

Input:
{json.dumps(payload, ensure_ascii=False)}

Output format:
[
  {{
    "id": 0,
    "text": "သဘာဝကျတဲ့ မြန်မာစကား"
  }}
]
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    result = safe_json_loads(response.text)

    if not isinstance(result, list):
        raise RuntimeError("Translation JSON format မမှန်ပါ။")

    mapped = {}

    for item in result:
        try:
            idx = int(item.get("id"))
            text = str(item.get("text", "")).strip()

            if text:
                mapped[idx] = text

        except Exception:
            continue

    output = []

    for i, seg in enumerate(batch):
        new_seg = dict(seg)
        new_seg["text"] = mapped.get(i, seg["original"])
        output.append(new_seg)

    return output


def translate_all(client, segments):
    translated = []

    for i in range(0, len(segments), MAX_TRANSLATION_BATCH):
        batch = segments[i:i + MAX_TRANSLATION_BATCH]

        translated.extend(
            translate_batch(
                client,
                batch
            )
        )

    return translated


# ============================================================
# BUILD DUBBED AUDIO TIMELINE
# ============================================================

def build_dubbed_audio(
    segments,
    tts_dir,
    output_audio,
    video_duration,
    voice,
    progress_callback=None
):
    """
    Create one full-length dubbed audio timeline.

    Video timeline is NOT changed.
    """

    tts_dir.mkdir(parents=True, exist_ok=True)

    audio_parts = []

    cursor = 0.0

    total = len(segments)

    for index, seg in enumerate(segments):
        start = max(0.0, float(seg["start"]))
        end = min(video_duration, float(seg["end"]))

        if end <= start:
            continue

        target_duration = end - start

        # Silence before dialogue
        gap = start - cursor

        if gap > 0.03:
            silence_file = tts_dir / f"silence_{index:05d}.m4a"

            create_silence(
                silence_file,
                gap
            )

            audio_parts.append(silence_file)

        raw_tts = tts_dir / f"raw_{index:05d}.mp3"
        fitted_tts = tts_dir / f"fit_{index:05d}.m4a"

        generate_tts(
            seg["text"],
            raw_tts,
            voice
        )

        fit_audio_to_duration(
            raw_tts,
            fitted_tts,
            target_duration
        )

        audio_parts.append(fitted_tts)

        cursor = end

        if progress_callback:
            progress_callback(
                index + 1,
                total
            )

    # Fill remaining video duration
    remaining = video_duration - cursor

    if remaining > 0.03:
        final_silence = tts_dir / "final_silence.m4a"

        create_silence(
            final_silence,
            remaining
        )

        audio_parts.append(final_silence)

    if not audio_parts:
        raise RuntimeError("Dubbed audio parts မထွက်ပါ။")

    concat_file = tts_dir / "audio_concat.txt"

    with open(concat_file, "w", encoding="utf-8") as f:
        for item in audio_parts:
            path = str(item.resolve()).replace("'", "'\\''")
            f.write(f"file '{path}'\n")

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "24000",
            "-ac", "1",
            "-t", f"{video_duration:.3f}",
            str(output_audio)
        ],
        timeout=600
    )


# ============================================================
# FINAL VIDEO
# ============================================================

def create_final_video(input_video, dubbed_audio, output_video):
    """
    Keep original video stream.
    Remove original audio.
    Add Burmese dubbed audio.
    """

    run_cmd(
        [
            "ffmpeg",
            "-y",

            "-i", str(input_video),
            "-i", str(dubbed_audio),

            "-map", "0:v:0",
            "-map", "1:a:0",

            "-c:v", "copy",

            "-c:a", "aac",
            "-b:a", "128k",

            "-shortest",

            "-movflags", "+faststart",

            str(output_video)
        ],
        timeout=1200
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_environment():
    missing = []

    if not check_binary("ffmpeg"):
        missing.append("ffmpeg")

    if not check_binary("ffprobe"):
        missing.append("ffprobe")

    if missing:
        raise RuntimeError(
            "FFmpeg မတွေ့ပါ။ packages.txt ထဲမှာ ffmpeg ထည့်ထားတာ သေချာပါစေ။"
        )


# ============================================================
# UI CONTROLS
# ============================================================

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "🎬 Video Upload",
        type=["mp4", "mkv", "mov", "webm"],
        help="Movie/video file ကိုရွေးပါ။"
    )

with col2:
    voice_choice = st.selectbox(
        "🎙️ မြန်မာ Voice",
        [
            "👨 သီဟ — အမျိုးသား",
            "👩 နီလာ — အမျိုးသမီး"
        ]
    )

voice_code = (
    VOICE_THIHA
    if "သီဟ" in voice_choice
    else VOICE_NILAR
)


style_choice = st.selectbox(
    "🎭 Translation Style",
    [
        "🎬 သဘာဝကျတဲ့ ဇာတ်လမ်းပြောသူ",
        "🎥 Movie Recap Style",
        "🗣️ ရိုးရိုးနားလည်လွယ်တဲ့ မြန်မာစကား"
    ]
)


if uploaded_file:

    st.info(
        f"📁 {uploaded_file.name}  •  "
        f"{uploaded_file.size / (1024 * 1024):.2f} MB"
    )


start_button = st.button(
    "🚀 START AI DUBBING",
    disabled=(uploaded_file is None)
)


# ============================================================
# MAIN PIPELINE
# ============================================================

if start_button:

    if st.session_state.processing:
        st.warning("Processing လုပ်နေဆဲပါ။")
        st.stop()

    st.session_state.processing = True
    st.session_state.completed = False
    st.session_state.video_bytes = None
    st.session_state.script = ""
    st.session_state.segments = []
    st.session_state.last_error = ""

    started_at = time.time()

    st.session_state.progress_bar = st.progress(0)
    st.session_state.status_box = st.empty()

    work_dir = WORK_ROOT / str(int(time.time()))

    audio_dir = work_dir / "tts"

    try:
        validate_environment()

        work_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        audio_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        input_video = work_dir / "input_video"

        extracted_audio = work_dir / "source_audio.mp3"

        dubbed_audio = work_dir / "burmese_dubbed.m4a"

        output_video = work_dir / "final_burmese_dub.mp4"

        # ----------------------------------------------------
        # 1. SAVE VIDEO
        # ----------------------------------------------------

        set_status(
            "1/7 🎬 Video",
            "Video ကို workspace ထဲသိမ်းနေပါတယ်...",
            5,
            started_at
        )

        with open(input_video, "wb") as f:
            f.write(uploaded_file.getbuffer())

        video_duration = get_duration(input_video)

        if video_duration <= 0:
            raise RuntimeError(
                "Video duration မဖတ်နိုင်ပါ။ File ပျက်နေနိုင်ပါတယ်။"
            )

        # ----------------------------------------------------
        # 2. AUDIO EXTRACTION
        # ----------------------------------------------------

        set_status(
            "2/7 🎧 Audio",
            f"Audio ထုတ်နေပါတယ်... "
            f"Video duration {format_time(video_duration)}",
            12,
            started_at
        )

        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-i", str(input_video),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "mp3",
                "-b:a", "64k",
                str(extracted_audio)
            ],
            timeout=600
        )

        # ----------------------------------------------------
        # GEMINI CLIENT
        # ----------------------------------------------------

        api_key = st.secrets.get(
            "GEMINI_API_KEY",
            ""
        ).strip().strip('"').strip("'")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY မတွေ့ပါ။ "
                "Streamlit Secrets ထဲမှာ ထည့်ထားပါ။"
            )

        client = genai.Client(
            api_key=api_key
        )

        # ----------------------------------------------------
        # 3. TRANSCRIPT
        # ----------------------------------------------------

        set_status(
            "3/7 📝 Transcript",
            "Gemini က dialogue နဲ့ timestamp တွေထုတ်နေပါတယ်...",
            20,
            started_at
        )

        segments = generate_transcript(
            client,
            extracted_audio
        )

        # Remove segments outside video
        valid_segments = []

        for seg in segments:

            seg["start"] = max(
                0.0,
                min(float(seg["start"]), video_duration)
            )

            seg["end"] = max(
                seg["start"],
                min(float(seg["end"]), video_duration)
            )

            if seg["end"] - seg["start"] >= 0.10:
                valid_segments.append(seg)

        segments = valid_segments

        if not segments:
            raise RuntimeError(
                "အသုံးပြုနိုင်တဲ့ dialogue မတွေ့ပါ။"
            )

        st
