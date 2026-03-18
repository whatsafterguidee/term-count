"""
=============================================================
  Word Frequency Analyzer สำหรับนักแปล (อัปเกรด)
  ─────────────────────────────────────
  รองรับ: .txt, .docx, .pdf, .srt | วิเคราะห์ภาษาไทย/อังกฤษ 
  ฟีเจอร์: N-grams, KWIC, Word Cloud, Export CSV
=============================================================
"""

import re
import io
import collections

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ─── ติดตั้ง dependency พื้นฐานผ่าน requirements.txt บน Streamlit Cloud ───
from docx import Document
import PyPDF2
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from wordcloud import WordCloud

# ─── Stopwords ────────────────────────────────────────────────────────
DEFAULT_EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "up", "about", "into", "through",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "need", "dare", "ought",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "his", "her", "its", "they", "them", "their", "this", "that",
    "these", "those", "who", "which", "what", "how", "when", "where",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "as", "such", "while", "than", "then", "also", "just", "more",
    "s", "t", "re", "ve", "ll", "d", "m"
}

DEFAULT_TH_STOPWORDS = set(thai_stopwords())

# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Word Frequency Analyzer", page_icon="📖", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background: #0f1117; color: #e8e3d9; }
[data-testid="stSidebar"] { background: #16191f; border-right: 1px solid #2a2d35; }
h1, h2, h3 { font-family: 'Georgia', serif; }
hr { border-color: #2a2d35 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS (File Reading)
# ═══════════════════════════════════════════════════════════════════
def extract_text(file_bytes: bytes, ext: str) -> str:
    if ext == "txt":
        try: return file_bytes.decode("utf-8")
        except: return file_bytes.decode("latin-1")
    elif ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in doc.paragraphs)
    elif ext == "pdf":
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif ext == "srt":
        text = file_bytes.decode("utf-8", errors="ignore")
        # ลบ timestamp และหมายเลขบรรทัดของซับไตเติล
        text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'<[^>]+>', '', text) # ลบ HTML tags บางชนิด
        return text
    return ""

# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS (NLP & Analysis)
# ═══════════════════════════════════════════════════════════════════
def tokenize_and_ngram(text: str, lang: str, n_gram: int) -> list[str]:
    if lang == "English":
        text = text.lower()
        tokens = re.findall(r"[a-z]+", text)
    else: # Thai
        # กรองเอาเฉพาะตัวอักษรไทย อังกฤษ และตัวเลข
        text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', ' ', text)
        raw_tokens = word_tokenize(text, engine="newmm")
        tokens = [t.strip() for t in raw_tokens if t.strip()]
    
    # ทำกลุ่มคำ (N-grams)
    if n_gram > 1:
        ngrams = zip(*[tokens[i:] for i in range(n_gram)])
        tokens = [" ".join(ngram) for ngram in ngrams]
        
    return tokens

def count_words(tokens: list[str], stopwords: set, min_len: int, n_gram: int) -> pd.DataFrame:
    if n_gram == 1:
        filtered = [w for w in tokens if w not in stopwords and len(w.replace(" ", "")) >= min_len]
    else:
        # ถ้าเป็น n-gram จะกรองออกก็ต่อเมื่อทุกคำในกลุ่มเป็น stopword (ปรับ logic ตามเหมาะสม)
        filtered = [w for w in tokens if not all(sub_w in stopwords for sub_w in w.split())]

    counter = collections.Counter(filtered)
    df = pd.DataFrame(counter.most_common(), columns=["คำ/กลุ่มคำ", "จำนวนครั้ง"])
    df.index = df.index + 1
    return df

def generate_kwic(text: str, keyword: str, window: int = 40):
    """ค้นหาคำที่ต้องการและดึงบริบทรอบข้างมาแสดง (Key Word In Context)"""
    results = []
    # ใช้ regex หาคำแบบไม่สนใจตัวพิมพ์เล็ก-ใหญ่
    for match in re.finditer(re.escape(keyword), text, re.IGNORECASE):
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        left_context = text[start:match.start()].replace('\n', ' ')
        right_context = text[match.end():end].replace('\n', ' ')
        results.append((left_context, match.group(), right_context))
    return pd.DataFrame(results, columns=["บริบทก่อนหน้า (Left Context)", "คำค้นหา (Keyword)", "บริบทตามหลัง (Right Context)"])

# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ การตั้งค่า")
    
    # ── เลือกภาษา ──
    lang = st.radio("ภาษาของเอกสาร", ["English", "ภาษาไทย (Thai)"])
    base_stopwords = DEFAULT_TH_STOPWORDS if lang == "ภาษาไทย (Thai)" else DEFAULT_EN_STOPWORDS
    
    # ── N-gram ──
    n_gram = st.radio("วิเคราะห์แบบกลุ่มคำ", [1, 2, 3], format_func=lambda x: "คำเดี่ยว (1-gram)" if x==1 else f"กลุ่มคำติดกัน ({x}-grams)")
    
    st.divider()
    top_n = st.slider("จำนวนคำที่แสดง (Top N)", 5, 50, 20, 5)
    min_len = st.slider("ความยาวคำขั้นต่ำ (ตัวอักษร)", 1, 10, 2)
    
    st.divider()
    st.markdown("**เพิ่ม Stopwords ของคุณเอง**")
    extra_sw_input = st.text_area("คั่นด้วยช่องว่าง", placeholder="e.g. ครับ ค่ะ")
    extra_stopwords = set(extra_sw_input.lower().split())
    all_stopwords = base_stopwords | extra_stopwords
    st.caption(f"Stopwords ในระบบ: {len(all_stopwords)} คำ")

# ═══════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ═══════════════════════════════════════════════════════════════════
st.title("📖 Word Frequency & Context Analyzer")
st.markdown("รองรับไฟล์: **.txt, .docx, .pdf, .srt** | วิเคราะห์คำเดี่ยว กลุ่มคำ และบริบทแวดล้อม")
st.divider()

uploaded_file = st.file_uploader("📂 อัปโหลดเอกสารของคุณ", type=["txt", "docx", "pdf", "srt"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

    with st.spinner("กำลังอ่านและวิเคราะห์เอกสาร..."):
        raw_text = extract_text(file_bytes, ext)
        tokens = tokenize_and_ngram(raw_text, lang.split(" ")[0], n_gram)
        df_all = count_words(tokens, all_stopwords, min_len, n_gram)

    # ── Metric ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 ชนิดไฟล์", ext.upper())
    col2.metric("🔤 จำนวน Token ทั้งหมด", f"{len(tokens):,}")
    col3.metric("✂️ หลังกรองคำทั่วไป", f"{df_all['จำนวนครั้ง'].sum():,}")
    col4.metric("🗂️ คำ/กลุ่มคำ ไม่ซ้ำ", f"{len(df_all):,}")
    st.divider()

    if df_all.empty:
        st.warning("ไม่พบคำในเอกสาร หรือทุกคำถูกกรองออกโดย stopwords")
    else:
        # ── TABS FOR VISUALIZATION ──
        tab1, tab2, tab3 = st.tabs(["📋 ตารางสถิติ & กราฟ", "☁️ Word Cloud (ภาพรวมคำศัพท์)", "🔎 KWIC (ดูบริบทแวดล้อมประโยค)"])
        
        with tab1:
            col_chart, col_table = st.columns([1.2, 1])
            df_display = df_all.head(top_n).copy()
            
            with col_chart:
                fig, ax = plt.subplots(figsize=(6, max(4, top_n * 0.3)))
                fig.patch.set_facecolor("#1c1f29")
                ax.set_facecolor("#1c1f29")
                # สำหรับภาษาไทยบนกราฟ Matplotlib ใน Streamlit อาจจะเหลื่อมถ้าไม่มีฟอนต์ 
                # แต่ภาษาอังกฤษแสดงผลปกติ
                data = df_display.iloc[::-1]
                bars = ax.barh(data["คำ/กลุ่มคำ"], data["จำนวนครั้ง"], color="#e8b84b", height=0.6)
                ax.tick_params(colors="#c8c2b4")
                for spine in ax.spines.values(): spine.set_visible(False)
                st.pyplot(fig)
                plt.close(fig)

            with col_table:
                st.dataframe(df_display, use_container_width=True, height=450)
                csv = df_all.to_csv(index=True, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("⬇️ ดาวน์โหลด CSV ทั้งหมด", csv, "frequency_export.csv", "text/csv")

        with tab2:
            st.markdown("#### แผนภาพคลาวด์คำศัพท์")
            freq_dict = dict(zip(df_all["คำ/กลุ่มคำ"], df_all["จำนวนครั้ง"]))
            try:
                # หมายเหตุ: หากใช้งานภาษาไทยและเกิดกล่องสี่เหลี่ยม ให้นำไฟล์ .ttf (เช่น THSarabunNew.ttf) มาใส่ 
                # แล้วเปลี่ยน font_path="THSarabunNew.ttf"
                font_p = "THSarabunNew.ttf" if lang == "ภาษาไทย (Thai)" else None 
 
                wc = WordCloud(width=800, height=400, background_color="#1c1f29", colormap="Wistia", font_path=font_p)
                wc.generate_from_frequencies(freq_dict)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                fig_wc.patch.set_facecolor("#1c1f29")
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
                plt.close(fig_wc)
                if lang == "ภาษาไทย (Thai)":
                    st.caption("*หากคำภาษาไทยแสดงเป็นกล่องสี่เหลี่ยม กรุณาตรวจสอบการตั้งค่าไฟล์ Font (.ttf) ในโปรเจกต์ของคุณ*")
            except Exception as e:
                st.error(f"ไม่สามารถสร้าง Word Cloud ได้ (อาจเกิดจากไม่มีไฟล์ฟอนต์ภาษาไทย): {e}")

        with tab3:
            st.markdown("#### ระบบวิเคราะห์บริบทแวดล้อม (Key Word In Context)")
            st.write("พิมพ์คำศัพท์ที่คุณต้องการดูตัวอย่างการใช้งานในประโยค (พิมพ์จากตารางด้านซ้ายก็ได้)")
            
            search_term = st.text_input("🔍 พิมพ์คำค้นหา:", "")
            context_window = st.slider("ความกว้างของบริบท (ตัวอักษร)", 20, 150, 50)
            
            if search_term:
                kwic_df = generate_kwic(raw_text, search_term, context_window)
                if not kwic_df.empty:
                    st.success(f"พบคำว่า '{search_term}' จำนวน {len(kwic_df)} ครั้งในเอกสาร")
                    st.dataframe(kwic_df, use_container_width=True)
                else:
                    st.warning("ไม่พบคำนี้ในเอกสารต้นฉบับ")
else:
    st.info("กรุณาอัปโหลดไฟล์ (.txt, .docx, .pdf, หรือ .srt) เพื่อเริ่มการวิเคราะห์")
