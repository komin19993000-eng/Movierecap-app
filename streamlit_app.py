import os
import time
import json
import asyncio
import subprocess
import streamlit as st
from google import genai
from google.genai import types
import edge_tts

# UI Setup
st.set_page_config(page_title="AI Movie Recap Automator", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d0f12; color: #e0e6ed; }
    h1 { color: #00f2fe !important; font-weight: 800; text-shadow: 0px 0px 10px rgba(0,242,254,0.3); }
    .stButton>button {
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        color: #000000 !important; font-weight: bold; border: none;
        border-radius: 8px; padding: 14px 28px; width: 100%; font-size: 16px;
    }
    .stProgress > div > div > div > div { background-color: #00f2fe; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ AI Movie Recap Automator (Mid-point Sync Core)")
st.write("Upload video to generate balanced dialogue-synced Burmese recaps.")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")

if "recap_complete" not in st.session_state:
    st.session_state.recap_complete = False
if "video_bytes" not in st.session_state:
    st.session_state.video_bytes = None
if "burmese_script" not in st.session_state:
    st.session_state.burmese_script = ""

async def generate_tts(text, output_file, voice_name):
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(output_file)

def get_media_duration(file_path):
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprintwrappers=1:nokey=1 \"{file_path}\""
    try:
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        return float(output)
    except:
        return 0.0

with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("🎬 Upload Video File (MP4, MKV, MOV)", type=["mp4", "mkv", "mov"])
    with col2:
        voice_choice = st.selectbox(
            "🎙️ Voiceover Voice Selection",
            options=["အမျိုးသား (သီဟ)", "အမျိုးသမီး (နီလာ)"]
        )
        voice_code = "my-MM-ThihaNeural" if "သီဟ" in voice_choice else "my-MM-NilarNeural"

if uploaded_file and st.button("🚀 Start Recap Generation Process"):
    if not GEMINI_API_KEY:
        st.error("🔑 Streamlit Secrets ထဲတွင် GEMINI_API_KEY မရှိသေးပါ။")
        st.stop()

    st.session_state.recap_complete = False
    st.session_state.video_bytes = None

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    work_dir = "temp_workspace"
    os.makedirs(work_dir, exist_ok=True)
    input_video_path = os.path.join(work_dir, "input_video.mp4")
    extracted_audio_path = os.path.join(work_dir, "extracted_audio.mp3")
    output_video_path = os.path.join(work_dir, "output_recap.mp4")

    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Step 1: Extract Audio
        status_text.markdown("### 🔊 Step 1/5: Extracting Audio...")
        progress_bar.progress(15)
        subprocess.run(["ffmpeg", "-y", "-i", input_video_path, "-q:a", "0", "-map", "a", extracted_audio_path], check=True)

        # Step 2: Gemini Timecode Extraction
        status_text.markdown("### 📝 Step 2/5: Processing Scene Timestamps & Script...")
        progress_bar.progress(35)
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        uploaded_audio = client.files.upload(file=extracted_audio_path)

        prompt = (
            "Listen to this audio and split it into dialogue segments with timestamps. "
            "Translate each segment into natural spoken Burmese narration. "
            "Return JSON format: list of objects with 'start' (float), 'end' (float), 'text' (Burmese narration)."
        )
        
        models_to_try = ["gemini-2.5-flash", "gemini-3.6-flash"]
        response = None
        
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[uploaded_audio, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                if response:
                    break
            except Exception:
                continue

        if not response:
            raise RuntimeError("API Connection Error! စက္ကန့်နည်းငယ်အကြာတွင် 'Start' ကို ပြန်နှိပ်ပေးပါ။")

        segments = json.loads(response.text)
        
        full_text_script = "\n".join([f"[{seg.get('start', 0)}s - {seg.get('end', 0)}s] {seg.get('text', '')}" for seg in segments])
        st.session_state.burmese_script = full_text_script

        # Step 3 & 4: Mid-point Speed Sync per Scene
        status_text.markdown("### ⏱️ Step 3 & 4/5: Applying Mid-point Sync per Dialogue...")
        progress_bar.progress(65)
        
        concat_list_file = os.path.join(work_dir, "concat_list.txt")
        
        with open(concat_list_file, "w") as cl_file:
            for idx, seg in enumerate(segments):
                start_t = seg.get("start", 0)
                end_t = seg.get("end", start_t + 2)
                v_dur = max(end_t - start_t, 0.5)
                text = seg.get("text", "").strip()
                
                if not text:
                    continue

                seg_video_raw = os.path.join(work_dir, f"raw_clip_{idx}.mp4")
                seg_tts_raw = os.path.join(work_dir, f"raw_tts_{idx}.mp3")
                seg_final = os.path.join(work_dir, f"out_{idx}.mp4")

                # 1. Trim Raw Video
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(start_t), "-i", input_video_path,
                    "-t", str(v_dur), "-c:v", "copy", "-an", seg_video_raw
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # 2. Generate Raw TTS
                asyncio.run(generate_tts(text, seg_tts_raw, voice_code))
                a_dur = get_media_duration(seg_tts_raw)

                if a_dur <= 0:
                    a_dur = v_dur

                # 3. Calculate Mid-point Target Duration & Speed Ratios
                target_dur = (v_dur + a_dur) / 2.0
                
                video_speed_factor = v_dur / target_dur
                audio_speed_factor = a_dur / target_dur

                # Bound factors between 0.7x and 1.3x for natural quality
                video_speed_factor = max(0.7, min(video_speed_factor, 1.3))
                audio_speed_factor = max(0.7, min(audio_speed_factor, 1.3))

                # 4. Render Scene Clip with Mid-point Speed Filters
                pts_filter = f"setpts={1/video_speed_factor}*PTS"
                atempo_filter = f"atempo={audio_speed_factor}"

                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", seg_video_raw,
                    "-i", seg_tts_raw,
                    "-filter_complex",
                    f"[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,{pts_filter}[v];"
                    f"[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,{atempo_filter}[a]",
                    "-map", "[v]",
                    "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest", seg_final
                ]
                
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                cl_file.write(f"file '{os.path.abspath(seg_final)}'\n")

        # Step 5: Merge Final Clips
        status_text.markdown("### 🎬 Step 5/5: Finalizing Mid-point Synced Video...")
        progress_bar.progress(90)
        
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_file,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-movflags", "+faststart", output_video_path
        ], check=True)

        if os.path.exists(output_video_path):
            with open(output_video_path, "rb") as f:
                st.session_state.video_bytes = f.read()
            st.session_state.recap_complete = True
            progress_bar.progress(100)
            status_text.markdown("✅ **Mid-point Dialogue Sync အောင်မြင်စွာ Render လုပ်ပြီးပါပြီ!**")

    except Exception as e:
        progress_bar.progress(0)
        status_text.empty()
        st.error(f"Error Details: {str(e)}")

# Display & Download
if st.session_state.recap_complete and st.session_state.video_bytes:
    st.markdown("---")
    st.subheader("📜 Generated Scene Dialogue Timestamps & Script:")
    st.text_area("Timestamps", st.session_state.burmese_script, height=200)

    st.video(st.session_state.video_bytes, format="video/mp4")

    st.download_button(
        label="📥 Download Dialogue-Synced Recap Video",
        data=st.session_state.video_bytes,
        file_name="movie_recap_synced.mp4",
        mime="video/mp4"
    )
