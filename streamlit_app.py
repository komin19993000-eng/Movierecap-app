import os
import time
import asyncio
import subprocess
import streamlit as st
from google import genai
import edge_tts

# UI Setup (အချက် - ၉: UI ဒီဇိုင်း ဆန်းသစ် လှပစေရန်)
st.set_page_config(page_title="AI Movie Recap Automator", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d0f12; color: #e0e6ed; }
    h1 { color: #00f2fe !important; font-weight: 800; text-shadow: 0px 0px 10px rgba(0,242,254,0.3); }
    .card {
        background: #1a1f26; border-radius: 12px; padding: 20px;
        border: 1px solid #2a323d; margin-bottom: 20px;
    }
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

# Audio Generation Helper
async def generate_tts(text, output_file, voice_name):
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(output_file)

# UI Layout Input Section
with st.container():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("🎬 Upload Video File (MP4, MKV, MOV)", type=["mp4", "mkv", "mov"])
        
    with col2:
        # အချက် - ၃: အမျိုးသား (သီဟ) သို့မဟုတ် အမျိုးသမီး (နီလာ) ကြိုတင်ရွေးချယ်ရန်
        voice_choice = st.selectbox(
            "🎙️ Voiceover Voice Selection",
            options=["အမျိုးသား (သီဟ)", "အမျိုးသမီး (နီလာ)"]
        )
        voice_code = "my-MM-ThihaNeural" if "သီဟ" in voice_choice else "my-MM-NilarNeural"

# အချက် - ၇: Start Processing နှိပ်သည်နှင့် လုပ်ငန်းစဉ်များ အလိုအလျောက် စတင်ခြင်း
if uploaded_file and st.button("🚀 Start Recap Generation Process"):
    if not GEMINI_API_KEY:
        st.error("🔑 Streamlit Secrets ထဲတွင် GEMINI_API_KEY မရှိသေးပါ။")
        st.stop()

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
        # အချက် - ၆: Step တိုင်းတွင် % နှင့် ခန့်မှန်း ကြာချိန် (ETA) ပြသခြင်း
        # Step 1: Extract Audio
        status_text.markdown("### 🔊 Step 1/5: Extracting Audio from Video...")
        progress_bar.progress(15)
        eta_text.info("⏱️ ခန့်မှန်း ကြာချိန်: ~၁၅ စက္ကန့်")
        
        subprocess.run(["ffmpeg", "-y", "-i", input_video_path, "-q:a", "0", "-map", "a", extracted_audio_path], check=True)

        # Step 2 & 8: AI Transcription & Model Auto-selection
        status_text.markdown("### 📝 Step 2/5: Transcribing Original Dialogue...")
        progress_bar.progress(35)
        eta_text.info("⏱️ ခန့်မှန်း ကြာချိန်: ~၂၀ စက္ကန့်")
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        uploaded_audio = client.files.upload(file=extracted_audio_path)

        # အချက် - ၁ & ၂: Transcript ထုတ်ယူပြီး သဘာဝကျသော ဇာတ်လမ်းပြောပြသူ စတိုင် ဘာသာပြန်ခြင်း
        prompt = (
            "1. Transcribe the audio precisely.\n"
            "2. Translate and summarize it into natural Burmese narration/storyteller style "
            "(ဇာတ်လမ်းပြောပြသူစတိုင် သဘာဝကျကျ ရေးသားပေးပါ။)."
        )
        
        # အချက် - ၈: မှန်ကန်သော Model Name အသုံးပြုခြင်း
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[uploaded_audio, prompt]
        )
        burmese_script = response.text

        # Step 3: Voiceover Generation
        status_text.markdown(f"### 🎙️ Step 3/5: Generating Burmese Voiceover ({voice_choice})...")
        progress_bar.progress(60)
        eta_text.info("⏱️ ခန့်မှန်း ကြာချိန်: ~၁၀ စက္ကန့်")
        
        asyncio.run(generate_tts(burmese_script, final_audio_path, voice_code))

        # အချက် - ၅: Audio နှင့် Video Timing ချိန်ညှိခြင်း (Atempo Adjustment)
        status_text.markdown("### ⏱️ Step 4/5: Aligning Audio & Video Timing...")
        progress_bar.progress(80)
        eta_text.info("⏱️ ခန့်မှန်း ကြာချိန်: ~၁၀ စက္ကန့်")

        cmd_v_len = f"ffprobe -v error -show_entries format=duration -of default=noprintwrappers=1:nokey=1 {input_video_path}"
        vid_len = float(subprocess.check_output(cmd_v_len, shell=True).decode().strip())

        cmd_a_len = f"ffprobe -v error -show_entries format=duration -of default=noprintwrappers=1:nokey=1 {final_audio_path}"
        aud_len = float(subprocess.check_output(cmd_a_len, shell=True).decode().strip())

        adjusted_audio_path = os.path.join(work_dir, "adjusted_audio.mp3")
        
        if aud_len > 0 and vid_len > 0:
            speed_ratio = aud_len / vid_len
            speed_ratio = max(0.8, min(speed_ratio, 1.25))
            subprocess.run([
                "ffmpeg", "-y", "-i", final_audio_path,
                "-filter:a", f"atempo={speed_ratio}", adjusted_audio_path
            ], check=True)
        else:
            adjusted_audio_path = final_audio_path

        # အချက် - ၄: မူရင်းအသံဖျောက်ပြီး အသံဖိုင်သစ် ပေါင်းစပ်ခြင်း
        status_text.markdown("### 🎬 Step 5/5: Merging Final Audio with Video...")
        progress_bar.progress(95)
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_video_path, "-i", adjusted_audio_path,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_video_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        # Finish Processing
        elapsed_time = round(time.time() - start_time, 2)
        progress_bar.progress(100)
        status_text.markdown("✅ **လုပ်ငန်းစဉ် အစအဆုံး ပြီးမြောက်ပါပြီ!**")
        eta_text.success(f"⚡ စုစုပေါင်း ကြာချိန်: {elapsed_time} စက္ကန့်")

        st.markdown("---")
        st.subheader("📜 Generated Burmese Script:")
        st.write(burmese_script)

        # အချက် - ၁၀: Download လုပ်နိုင်ခြင်း
        st.video(output_video_path)
        with open(output_video_path, "rb") as file:
            st.download_button(
                label="📥 Download Recap Video",
                data=file,
                file_name="movie_recap_final.mp4",
                mime="video/mp4"
            )

    except Exception as e:
        progress_bar.progress(0)
        status_text.empty()
        eta_text.empty()
        st.error(f"Error Details: {str(e)}")
