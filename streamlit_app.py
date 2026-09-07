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

st.title("⚡ AI Movie Recap Automator (Fast Dialogue Sync)")
st.write("Upload video to generate scene-by-scene frame-synced Burmese recaps.")

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

def get_audio_duration(file_path):
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

    start_time = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()
    eta_text = st.empty()
    
    work_dir = "temp_workspace"
    os.makedirs(work_dir, exist_ok=True)
    input_video_path = os.path.join(work_dir, "input_video.mp4")
    extracted_audio_path = os.path.join(work_dir, "extracted_audio.mp3")
    output_video_path = os.path.join(work_dir, "output_recap.mp4")

    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Step 1: Fast Audio Extraction
        status_text.markdown("### 🔊 Step 1/5: Extracting Audio...")
        progress_bar.progress(15)
        subprocess.run(["ffmpeg", "-y", "-i", input_video_path, "-q:a", "0", "-map", "a", extracted_audio_path], check=True)

        # Step 2: Optimized Gemini Prompt
        status_text.markdown("### 📝 Step 2/5: Processing Timestamps & Script...")
        progress_bar.progress(35)
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        uploaded_audio = client.files.upload(file=extracted_audio_path)

        prompt = (
            "Listen to this audio and split it into dialogue segments. "
            "Translate each segment into natural spoken Burmese narration. "
            "Return JSON format: list of objects with 'start' (float), 'end' (float), 'text' (Burmese narration)."
        )
        
        # Server ပိတ်ဆို့မှုမဖြစ်စေဘဲ အမြန်ဆုံး မော်ဒယ်များ သို့ ဦးစားပေးချိတ်ဆက်ခြင်း
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
            raise RuntimeError("Google API လိုင်း ခေတ္တကျနေပါသည်။ စက္ကန့်နည်းငယ်အကြာတွင် 'Start' ကို ပြန်နှိပ်ပေးပါ။")

        segments = json.loads(response.text)
        
        full_text_script = "\n".join([f"[{seg.get('start', 0)}s - {seg.get('end', 0)}s] {seg.get('text', '')}" for seg in segments])
        st.session_state.burmese_script = full_text_script

        # Step 3 & 4: Dialogue Processing
        status_text.markdown("### ⏱️ Step 3 & 4/5: Syncing Audio & Video per Scene...")
        progress_bar.progress(65)
        
        concat_list_file = os.path.join(work_dir, "concat_list.txt")
        
        with open(concat_list_file, "w") as cl_file:
            for idx, seg in enumerate(segments):
                start_t = seg.get("start", 0)
                end_t = seg.get("end", start_t + 2)
                orig_dur = max(end_t - start_t, 0.5)
                text = seg.get("text", "")
                
                if not text.strip():
                    continue

                seg_video = os.path.join(work_dir, f"clip_{idx}.mp4")
                seg_tts = os.path.join(work_dir, f"tts_{idx}.mp3")
                seg_final = os.path.join(work_dir, f"out_{idx}.mp4")

                # Video Trim
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(start_t), "-i", input_video_path,
                    "-t", str(orig_dur), "-c:v", "libx264", "-an", seg_video
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Edge-TTS (မြန်မာအသံ)
                asyncio.run(generate_tts(text, seg_tts, voice_code))
                tts_dur = get_audio_duration(seg_tts)

                # Frame & Speed Syncing
                if tts_dur > 0:
                    pts_speed = orig_dur / tts_dur
                    pts_speed = max(0.6, min(pts_speed, 1.8))
                    
                    subprocess.run([
                        "ffmpeg", "-y", "-i", seg_video, "-i", seg_tts,
                        "-filter_complex", f"[0:v]setpts={1/pts_speed}*PTS[v]",
                        "-map", "[v]", "-map", "1:a:0", "-c:v", "libx264", "-c:a", "aac",
                        "-shortest", seg_final
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    seg_final = seg_video

                cl_file.write(f"file '{os.path.basename(seg_final)}'\n")

        # Step 5: Fast Merging
        status_text.markdown("### 🎬 Step 5/5: Finalizing Video...")
        progress_bar.progress(90)
        
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_file,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p", "-c:a", "aac", output_video_path
        ], check=True)

        if os.path.exists(output_video_path):
            with open(output_video_path, "rb") as f:
                st.session_state.video_bytes = f.read()
            st.session_state.recap_complete = True
            progress_bar.progress(100)
            status_text.markdown("✅ **အသံနှင့် ဗီဒီယို ကွက်တိကျစွာ Render လုပ်ပြီးပါပြီ!**")

    except Exception as e:
        progress_bar.progress(0)
        status_text.empty()
        eta_text.empty()
        st.error(f"Error Details: {str(e)}")

# Display & Download
if st.session_state.recap_complete and st.session_state.video_bytes:
    st.markdown("---")
    st.subheader("📜 Scene Dialogue Timestamps & Burmese Script:")
    st.text_area("Timestamps", st.session_state.burmese_script, height=200)

    st.video(st.session_state.video_bytes, format="video/mp4")

    st.download_button(
        label="📥 Download Dialogue-Synced Recap Video",
        data=st.session_state.video_bytes,
        file_name="movie_recap_synced.mp4",
        mime="video/mp4"
    )
