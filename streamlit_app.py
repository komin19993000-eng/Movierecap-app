import os
import asyncio
import subprocess
import streamlit as st
import google.generativeai as genai
import edge_tts

st.set_page_config(page_title="AI Movie Recap Automator", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d0f12; color: #e0e6ed; }
    h1, h2, h3 { color: #00f2fe !important; }
    .stButton>button {
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        color: #000000 !important; font-weight: bold; border: none; border-radius: 8px; padding: 12px 24px;
    }
    .stProgress > div > div > div > div { background-color: #00f2fe; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Auto Movie Recap Engine")
st.write("Upload video to automatically generate Burmese dub/voiceover recap.")

# Secrets ထဲက Key ကို တိုက်ရိုက်ဖတ်ယူခြင်း
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

async def generate_tts(text, output_file):
    communicate = edge_tts.Communicate(text, "my-MM-ThihaNeural")
    await communicate.save(output_file)

uploaded_file = st.file_uploader("🎬 Upload Video File (MP4, MKV, MOV)", type=["mp4", "mkv", "mov"])

if uploaded_file and st.button("🚀 Start Processing"):
    if not GEMINI_API_KEY:
        st.error("🔑 Streamlit Secrets ထဲတွင် GEMINI_API_KEY မရှိသေးပါ။ Manage App > Secrets တွင် ထည့်သွင်းပေးပါ။")
        st.stop()
        
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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
        status_text.markdown("**[Step 1/4]** 🔊 Audio ခွဲထုတ်နေပါသည်။")
        progress_bar.progress(25)
        subprocess.run(["ffmpeg", "-y", "-i", input_video_path, "-q:a", "0", "-map", "a", extracted_audio_path], check=True)

        # Step 2: Gemini Translation
        status_text.markdown("**[Step 2/4]** 🧠 Gemini AI မှ မြန်မာလို ဘာသာပြန်နေပါသည်။")
        progress_bar.progress(50)
        
        genai.configure(api_key=GEMINI_API_KEY)
        audio_file = genai.upload_file(path=extracted_audio_path)
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = "Listen to the dialogue and summarize/translate into natural Burmese movie recap script."
        response = model.generate_content([audio_file, prompt])
        burmese_script = response.text

        # Step 3: Voiceover (Edge-TTS)
        status_text.markdown("**[Step 3/4]** 🎙️ Voiceover အသံ ထုတ်လုပ်နေပါသည်။")
        progress_bar.progress(75)
        asyncio.run(generate_tts(burmese_script, final_audio_path))

        # Step 4: Merge Audio & Video
        status_text.markdown("**[Step 4/4]** 🎬 အသံနှင့် ဗီဒီယို ပေါင်းစပ်နေပါသည်။")
        progress_bar.progress(90)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_video_path, "-i", final_audio_path,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_video_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        progress_bar.progress(100)
        status_text.markdown("✅ **ပြီးမြောက်ပါပြီ!**")
        st.success("🎉 Video Recap ပြုလုပ်ခြင်း အောင်မြင်ပါသည်။")
        st.video(output_video_path)
        
        with open(output_video_path, "rb") as file:
            st.download_button("📥 Download Video", data=file, file_name="movie_recap.mp4", mime="video/mp4")

    except Exception as e:
        progress_bar.progress(0)
        status_text.empty()
        st.error(f"Error Details: {str(e)}")
