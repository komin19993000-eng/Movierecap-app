import streamlit as st
import google.generativeai as genai
import pysrt
import re
import io

# 1. API Keys စာရင်း (ကြိုက်သလောက် ထည့်ထားနိုင်သည်)
API_KEYS = [
    "YOUR_GEMINI_API_KEY_1",
    "YOUR_GEMINI_API_KEY_2",
    "YOUR_GEMINI_API_KEY_3",
]

# Session State ထဲမှာ Key Index နဲ့ Error Tracking ကို မှတ်ထားခြင်း
if "key_index" not in st.session_state:
    st.session_state.key_index = 0

def get_current_api_key():
    return API_KEYS[st.session_state.key_index]

def rotate_api_key():
    st.session_state.key_index = (st.session_state.key_index + 1) % len(API_KEYS)

# 2. Batch လိုက် ဘာသာပြန်ပေးည့် Function (Rate Limit မိရင် Key အလိုအလျောက် လဲမည်)
def translate_batch_with_retry(texts_batch):
    max_retries = len(API_KEYS)
    
    # တိုတိုတိုင်တိုင်နှင့် အဓိပ္ပာယ်မပျက်စေရန် ပြင်ဆင်ထားသော System Prompt
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
            # Rate Limit (429) သို့မဟုတ် API Key ပြဿနာတက်လျှင် Key နောက်တစ်ခုသို့ ပြောင်းမည်
            if "429" in str(e) or "quota" in str(e).lower() or "key" in str(e).lower():
                rotate_api_key()
            else:
                # အခြား Error ဖြစ်ပါက Key လဲပြီး ထပ်မံ ကြိုးစားမည်
                rotate_api_key()
                
    st.error("API Keys အားလုံး Limit ပြည့်သွားပါပြီ။ ကျေးဇူးပြု၍ ခဏစောင့်ပြီးမှ ပြန်လည်ကြိုးစားပါ။")
    return None

# 3. Streamlit UI
st.title("🇲🇲 Subtitle Translator & Dubbing Optimizer")

uploaded_file = st.file_uploader("SRT ဖိုင်တင်ပါ", type=["srt"])

if uploaded_file is not None:
    # SRT ဖိုင်ကို Read လုပ်ခြင်း
    srt_content = uploaded_file.getvalue().decode("utf-8")
    subs = pysrt.from_string(srt_content)
    
    if st.button("မြန်မာ SRT ထုတ်မယ်"):
        translated_subs = pysrt.SubRipFile()
        
        batch_size = 10
        total_subs = len(subs)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Batch လိုက် ခွဲခြား၍ ဘာသာပြန်ခြင်း ( Limit တက်လျှင် ရောက်သည့်နေရာမှ ဆက်သွားမည် )
        for i in range(0, total_subs, batch_size):
            batch = subs[i:i + batch_size]
            formatted_batch = [f"{j+1}: {item.text}" for j, item in enumerate(batch)]
            
            status_text.text(f"ဘာသာပြန်နေသည်... ({i}/{total_subs} lines)")
            
            translated_text = translate_batch_with_retry(formatted_batch)
            
            if translated_text is None:
                st.stop()
                
            # ပြန်လာသော Text များကို SRT Subtitle Item များအဖြစ် ပြန်လည် ထည့်သွင်းခြင်း
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
            
            # Progress bar မြှင့်ခြင်း
            progress_bar.progress(min((i + batch_size) / total_subs, 1.0))
            
        status_text.text(f"SRT ပြီးပါပြီ — {len(translated_subs)} subtitle lines")
        
        # Myanmar SRT Preview ပြသခြင်း
        st.subheader("Myanmar SRT Preview")
        preview_text = ""
        for item in translated_subs[:10]:  # ပထမ ၁၀ ကြောင်းကို Preview ပြမည်
            preview_text += f"{item.index}\n{item.start} --> {item.end}\n{item.text}\n\n"
        st.text_area("Preview", value=preview_text, height=250)
        
        # Download Button
        final_srt_string = translated_subs.text
        st.download_button(
            label="Download Myanmar SRT",
            data=final_srt_string,
            file_name="translated_myanmar.srt",
            mime="application/x-subrip"
        )
