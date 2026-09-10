import streamlit as st
import google.generativeai as genai
import pysrt
import re

# 1. API Keys စာရင်း
API_KEYS = [
    "YOUR_GEMINI_API_KEY_1",
    "YOUR_GEMINI_API_KEY_2",
    "YOUR_GEMINI_API_KEY_3",
]

if "key_index" not in st.session_state:
    st.session_state.key_index = 0

def get_current_api_key():
    return API_KEYS[st.session_state.key_index]

def rotate_api_key():
    st.session_state.key_index = (st.session_state.key_index + 1) % len(API_KEYS)

# 2. Batch ဘာသာပြန်ခြင်း (Rate Limit မိပါက ရောက်သည့်နေရာမှ ဆက်သွားမည်)
def translate_batch_with_retry(texts_batch):
    max_retries = len(API_KEYS)
    
    # စကားပြောဟန် တိုတိုတိုင်တိုင်ဖြစ်စေရန် ပြင်ဆင်ထားသော Prompt
    system_prompt = (
        "You are a professional translator for SRT subtitles to Myanmar spoken language.\n"
        "RULES:\n"
        "1. Keep the original meaning 100% intact, but make the translation as short and concise as possible for dubbing/TTS.\n"
        "2. Avoid unnecessary formal ending words (e.g., do NOT use 'ခဲ့ရပါတယ်', 'သွားခဲ့ရပါတယ်', 'ပါသည်', 'ဖြစ်ပါသည်။'). Use short natural spoken Burmese.\n"
        "3. Maintain exact numbered line matching for each item.\n"
        "4. Do not summarize or merge lines."
    )

    for attempt in range(max_retries):
        try:
            current_key = get_current_api_key()
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt_content = f"{system_prompt}\n\nTranslate the following numbered lines into concise spoken Myanmar:\n" + "\n".join(texts_batch)
            
            response = model.generate_content(prompt_content)
            return response.text
        except Exception as e:
            rotate_api_key()
                
    st.error("API Keys အားလုံး Limit ပြည့်သွားပါပြီ။ ကျေးဇူးပြု၍ ခဏစောင့်ပြီးမှ ပြန်လည်ကြိုးစားပါ။")
    return None

# 3. Custom CSS UI Styling (မူလ ဒီဇိုင်းအတိုင်း ပြန်လည်ထည့်သွင်းထားသည်)
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# 4. Streamlit App Interface
st.title("🇲🇲 Subtitle Translator & Dubbing Optimizer")

uploaded_file = st.file_uploader("SRT ဖိုင်တင်ပါ", type=["srt"])

if uploaded_file is not None:
    srt_content = uploaded_file.getvalue().decode("utf-8")
    subs = pysrt.from_string(srt_content)
    
    if st.button("📝 မြန်မာ SRT ထုတ်မယ်"):
        translated_subs = pysrt.SubRipFile()
        
        batch_size = 10
        total_subs = len(subs)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, total_subs, batch_size):
            batch = subs[i:i + batch_size]
            formatted_batch = [f"{j+1}: {item.text}" for j, item in enumerate(batch)]
            
            status_text.text(f"ဘာသာပြန်နေသည်... ({i}/{total_subs} lines)")
            
            translated_text = translate_batch_with_retry(formatted_batch)
            
            if translated_text is None:
                st.stop()
                
            lines = translated_text.strip().split("\n")
            translated_dict = {}
            for line in lines:
                match = re.match(r"^(\d+)[\.\:]\s*(.*)", line.strip())
                if match:
                    idx = int(match.group(1)) - 1
                    translated_dict[idx] = match.group(2)
            
            for j, sub_item in enumerate(batch):
                new_text = translated_dict.get(j, sub_item.text)
                new_sub = pysrt.SubRipItem(
                    index=sub_item.index,
                    start=sub_item.start,
                    end=sub_item.end,
                    text=new_text
                )
                translated_subs.append(new_sub)
            
            progress_bar.progress(min((i + batch_size) / total_subs, 1.0))
            
        status_text.success(f"✔ SRT ပြီးပါပြီ — {len(translated_subs)} subtitle lines")
        
        st.subheader("Myanmar SRT Preview")
        preview_text = ""
        for item in translated_subs[:10]:
            preview_text += f"{item.index}\n{item.start} --> {item.end}\n{item.text}\n\n"
        st.text_area("Preview", value=preview_text, height=250)
        
        final_srt_string = translated_subs.text
        st.download_button(
            label="⬇ Download Myanmar SRT",
            data=final_srt_string,
            file_name="translated_myanmar.srt",
            mime="application/x-subrip"
        )
