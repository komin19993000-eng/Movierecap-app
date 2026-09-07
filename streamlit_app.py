import os
import time
import asyncio
import subprocess
import streamlit as st
import google.genai as genai
import edge_tts

# 1. UI Configuration (Neon Dark Theme)
st.set_page_config(page_title="AI Movie Recap Automator", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d0f12; color: #e0e6ed; }
    h1, h2, h3 { color: #00f2fe !important; }
    .stButton>button {
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        color: #000000 !important;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
    }
    .stProgress > div > div > div > div { background-color: #00f2fe; }
    .error-box { background-color: #2a080c; border: 1px solid #ff4b4b; padding: 15px; border-radius: 8px; color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Auto Movie Recap Engine")
st.write("Upload video to automatically generate Burmese dub/voiceover recap.")

GEMINI_API_KEY = st.sidebar.text_input("Enter Gemini API Key", type="password")

def get_gemini_client():
    if not GEMINI_API_KEY:
        st.error("🔑 ကျေးဇူးပြု၍ Gemini API Key ထည့်သွင်းပေးပါ။")
        st.stop()
    return genai.Client(api_key=GEMINI_API_KEY)

def call_gemini_with_fallback(client, contents, prompt):
    models = ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
    last_err = None
    for m in models:
        try:
            response = client.models.generate_content(
                model=m,
                contents=[contents, prompt]
            )
            return response.text
        except Exception as e:
            last_err = e
            continue
    raise Exception(f"AI models error: {str(last_err)}")

async def generate_tts(text, output_file, voice="my-MM-ThihaNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

uploaded_file = st.file_uploader("🎬 Upload Video File (MP4, MKV, MOV)", type=["mp4", "mkv", "mov"])

if uploaded_file and st.button("🚀 Start Processing"):
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
        # Step 1
        status_text.markdown("**[Step 1/5 - 20%]** 🔊 Audio ခွဲထုတ်နေပါသည်။")
        progress_bar.progress(20)
        subprocess.run(["ffmpeg", "-y", "-i", input_video_path, "-q:a", "0", "-map", "a", extracted_audio_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 2
        status_text.markdown("**[Step 2/5 - 40%]** 🧠 Gemini AI မှ မြန်မာလို ဘာသာပြန်နေပါသည်။")
        progress_bar.progress(40)
        client = get_gemini_client()
        uploaded_audio_file = client.files.upload(file=extracted_audio_path)
        
        prompt = "Listen to dialogue and translate to natural Burmese movie recap style narrative text."
        burmese_script = call_gemini_with_fallback(client, uploaded_audio_file, prompt)

        # Step 3
        status_text.markdown("**[Step 3/5 - 60%]** 🎙️ Voiceover အသံ ထုတ်လုပ်နေပါသည်။")
        progress_bar.progress(60)
        asyncio.run(generate_tts(burmese_script, final_audio_path))

        # Step 4
        status_text.markdown("**[Step 4/5 - 80%]** 🎬 အသံအသစ်နှင့် ဗီဒီယို ပေါင်းစပ်နေပါသည်။")
        progress_bar.progress(80)
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", input_video_path,
            "-i", final_audio_path,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_video_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 5
        progress_bar.progress(100)
        status_text.markdown("**[Step 5/5 - 100%]** ✅ ပြီးမြောက်ပါပြီ။")
        st.success("🎉 Video Recap ပြုလုပ်ခြင်း အောင်မြင်ပါသည်။")
        st.video(output_video_path)
        
        with open(output_video_path, "rb") as file:
            st.download_button("📥 Download Video", data=file, file_name="movie_recap.mp4", mime="video/mp4")

    except Exception as e:
        progress_bar.progress(0)
        status_text.empty()
        st.markdown(f'<div class="error-box"><h4>❌ Error Occurred</h4><p>{str(e)}</p></div>', unsafe_allow_html=True)
