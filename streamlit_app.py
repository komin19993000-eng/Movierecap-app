import os
import time
import asyncio
import subprocess
import streamlit as st
from google import genai
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

st.title("⚡ AI Movie Recap Automator")
st.write("Upload video and configure voiceover settings to build automated Burmese recaps.")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")

if "recap_complete" not in st.session_state:
    st.session_state.recap_complete = False
if "video_data" not in st.session_state:
    st.session_state.video_data = None
if "burmese_script" not in st.session_state:
    st.session_state.burmese_script = ""

async def generate_tts(text, output_file, voice_name):
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(output_file)

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
    st.session_state.video_data = None

    start_time = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()
    eta_text = st.empty()
    
    work_dir = "temp_workspace"
    os.makedirs(work_dir, exist_ok=True)
    input_video_path = os.path.join(work_dir, "input_video.mp4")
    extracted_audio_path = os.path.join(work_dir, "extracted_audio.mp3")
    final_audio_path = os.path.join(work_dir, "final_dub.mp3")
    output_video_path = os.path.join(work_dir, "output_recap.mp4")

    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        # Step 1: Extract Audio
        status_text.markdown("### 🔊 Step 1/5: Extracting Audio from Video...")
        progress_bar.progress(15)
        eta_text.info("⏱️ ခန့်မှန်း ကြာချိန်: ~၁၀ စက္ကန့်")
        subprocess.run(["ffmpeg", "-y", "-i", input_video_path, "-q:a", "0", "-map", "a", extracted_audio_path], check=True)

        # Step 2: Gemini Prompt (မြန်မာစာသီးသန့် ထုတ်ပေးရန် ပြင်ဆင်ထားပါသည်)
        status_text.markdown("### 📝 Step 2/5: Generating Burmese Storyteller Script...")
        progress_bar.progress(35)
        eta_text.info("⏱️ ခန့်မှန်း ကြာချိန်: ~၁၅ စက္ကန့်")
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        uploaded_audio = client.files.upload(file=extracted_audio_path)

        prompt = (
            "Listen to this audio and write a natural Burmese movie recap narration. "
            "IMPORTANT: Output ONLY the final spoken Burmese narration script. "
            "Do NOT include original transcripts, English text, scene descriptions, labels, or notes. "
            "Provide ONLY the clean Burmese script ready to be read aloud."
        )
        
        max_retries = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=[uploaded_audio, prompt]
                )
                break
            except Exception as req_err:
                if "503" in str(req_err) and attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise req_err

        burmese_script = response.text.strip()
        st.session_state.burmese_script = burmese_script

        # Step 3: Voiceover Generation
        status_text.markdown(f"### 🎙️ Step 3/5: Generating Burmese Voiceover ({voice_choice})...")
        progress_bar.progress(60)
        asyncio.run(generate_tts(burmese_script, final_audio_path, voice_code))

        # Step 4: Align Audio
        status_text.markdown("### ⏱️ Step 4/5: Aligning Audio & Video Timing...")
        progress_bar.progress(80)
        adjusted_audio_path = os.path.join(work_dir, "adjusted_audio.mp3")
        subprocess.run(["ffmpeg", "-y", "-i", final_audio_path, "-filter:a", "atempo=1.0", adjusted_audio_path], check=True)

        # Step 5: Merge with Compression (File Size သေးငယ်ပြီး Download မြန်ဆန်စေရန်)
        status_text.markdown("### 🎬 Step 5/5: Compressing & Merging Video...")
        progress_bar.progress(95)
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_video_path, "-i", adjusted_audio_path,
            "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_video_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        with open(output_video_path, "rb") as f:
            st.session_state.video_data = f.read()
            
        st.session_state.recap_complete = True
        progress_bar.progress(100)
        status_text.markdown("✅ **လုပ်ငန်းစဉ် အစအဆုံး ပြီးမြောက်ပါပြီ!**")

    except Exception as e:
        progress_bar.progress(0)
        status_text.empty()
        eta_text.empty()
        st.error(f"Error Details: {str(e)}")

# Display & Fast Download
if st.session_state.recap_complete and st.session_state.video_data:
    st.markdown("---")
    st.subheader("📜 Generated Burmese Script:")
    st.write(st.session_state.burmese_script)

    st.video(st.session_state.video_data)

    st.download_button(
        label="📥 Download Recap Video (Optimized Size)",
        data=st.session_state.video_data,
        file_name="movie_recap_final.mp4",
        mime="video/mp4"
    )
