# ✅ 整合 FAISS RAG 到現有 app.py
import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai
import tarfile
import gzip
import shutil
import tempfile
from io import BytesIO
from biom import load_table

# ✅ FAISS RAG 套件
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ---------- RAG 設定 ----------
INDEX_FILE_PATH = "microbio_faiss_index"
PDF_PATH = "C:\\Users\\User\\Downloads\\Microbiology and Immunology Textbook of 2nd Edition ( PDFDrive ).pdf"

from langchain_community.embeddings import HuggingFaceEmbeddings


def load_or_create_faiss():
    embedding = HuggingFaceEmbeddings()

    if os.path.exists(INDEX_FILE_PATH):
        return FAISS.load_local(INDEX_FILE_PATH, embeddings=embedding, allow_dangerous_deserialization=True)

    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    texts = [chunk.page_content for chunk in chunks]
    vector_store = FAISS.from_texts(texts, embedding)
    vector_store.save_local(INDEX_FILE_PATH)
    return vector_store
    
# ✅ 初始化 Gemini
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY","")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")
chat = model.start_chat()

# ✅ FAISS 載入
vector_store = load_or_create_faiss()

# ✅ Prompt 模板與 UI 請見原始程式碼（不重複列出）
# ⚠️ 若要使用 RAG，需要插入一個 Retrieval 函數如下：

def retrieve_context(query: str, k: int = 5):
    results = vector_store.similarity_search(query, k=k)
    context_texts = [doc.page_content for doc in results]
    return "\n\n".join(context_texts)


def generate_llm_prompt(mode, file_contents):
    file_definitions = {
        "Heatmap": "所有樣本與所有微生物分類的統計矩陣。",
        "Sample Metadata": "樣本的基本資訊，例如採樣時間與部位。",
        "Samples Overview": "每個樣本的 QC 統計與摘要。",
        "Sample Taxon Report": "各樣本中檢出的微生物分類與數據。",
        "Combined Sample Taxon Results": "彙整所有樣本的微生物數據總表。",
        "Contig Summary Reports": "Contig 的 QC 統計與覆蓋率。",
        "Host Genes Count": "宿主轉錄表現統計。",
        "Combined Microbiome File": "樣本與微生物分類合併成 BIOM 格式。",
        "Consensus Genome": "下載多個共識基因組，可選擇分開或合併為單一檔案。",
        "Consensus Genome Overview": "共識基因組的品質控制（QC）指標（例如：基因組覆蓋率百分比、比對到的讀數、SNP 數量）及其他統計摘要",
        "Intermediate Output Files": "中間產出的分析文件，包括 BAM 對齊檔案、覆蓋率圖、QUAST 報告等內容。",
        "Antimicrobial Assistance Results": "包含抗藥性報告、完整的抗藥性指標、中間分析結果及 CARD RGI 工具的輸出。",
        "Combined AMR Results": "整合樣本中抗藥性基因的指標（如覆蓋率、深度）為單一報告。"
    }

    summary_lines = []

    # 🔹 加入檔案功能定義（依據主題）
    if mode in ["Metagenomics", "Consensus Genome", "Antimicrobial Resistance"]:
        summary_lines.append("📘 檔案功能定義：")
        for label, definition in file_definitions.items():
            summary_lines.append(f"- **{label}**：{definition}")
        summary_lines.append("")

    # 🔹 加入使用者上傳的 CSV 摘要內容
    for label, content in file_contents.items():
        summary_lines.append(f"📄 檔案: {label}\n內容摘要:\n{content}\n")

    # 🔹 新增：查詢向量庫相關背景知識（教科書）
    user_query = f"{mode} 報告解讀重點與臨床風險"
    context_text = retrieve_context(user_query)
    summary_lines.append(f"\n📚 教科書補充知識：\n{context_text}")

    prompt_template = TEMPLATE_MAP[mode]
    return prompt_template.format(csv_content="\n".join(summary_lines))




# Prompt 模板
TEMPLATE_MAP = {
    "Metagenomics": """
你是一位微生物基因體學專家。請根據使用者上傳的 IDSEQ Metagenomics CSV 資料進行臨床導向的整體解讀。這些資料可能包含：

- Sample Taxons Report：樣本中檢出的微生物與其 reads/rPM
- Combined Sample Taxon Results：彙整後所有樣本的微生物豐度
- Taxon Heatmap：多樣本與微生物的量化矩陣
- Sample Metadata：樣本採集來源、時間、樣本種類
- Samples Overview：每個樣本的品質控制（QC）與通過率
- Contig Summary Reports：微生物基因片段的覆蓋率與序列比對品質
- Host Genes Count：宿主基因的表現情形（可能與感染、免疫反應相關）
- Combined Microbiome File：微生物社群組成與元數據整合

請你根據上述所有可能出現的資訊，綜合分析樣本中：

1. 檢出量最高的微生物（依 reads 數或 rPM/ratio），推測其可能感染來源（環境、腸道常見菌、潛在病原體）。
2. 判斷這些微生物是否具臨床意義，是否為污染或背景菌（例如唾液、皮膚、口腔常見菌）。
3. 若出現 WHO 公告的高優先警示病原體，或近年被監控的罕見菌，請列為「臨床需注意」。
4. 從 Contig 統計、Host gene 表現與 QC 指標中推測樣本品質與解讀可靠性。
5. 若可行，指出是否為可能的共感染情形，或具特徵性菌群改變（如 dysbiosis）。

📌 分析結果請條列整理，重點聚焦於臨床判讀與樣本風險評估，不必逐項報表說明。

CSV 原始摘要如下：
{csv_content}
""",
    "Consensus Genome": """
你是病毒基因組分析專家，請根據 Consensus Genome 比對與 QC 統計，做出專業見解：

1. 比對到的病毒種類是否為常見感染源或特殊新型變異株？請列出對應 SNP 數與相似度、覆蓋率。
2. 分析該定序片段是否能構成完整基因組，或為部分定序片段（並註明是否具診斷意義）。
3. 若有突變點符合 VOC（variant of concern）或與重大國際通報有關，請提出預警與推論。

你的任務是根據 CSV 整合出臨床參考價值的分析，不是單純列出統計值。

CSV 原始摘要如下：
{csv_content}
""",  # 省略內容，保留原本
    "Antimicrobial Resistance": """
你是一位臨床感染科與抗藥性基因體分析專家，請根據下列 CSV（Combined AMR Results）進行風險導向的分析與臨床見解判斷：

1. 從資料中找出樣本中出現的抗藥性基因（如 Erm(K)、TEM-116、OXA-780）與其對應的抗生素類型。
2. 判斷是否為 MDR（多重抗藥性）或 XDR（廣泛抗藥性）菌株，請依據國際標準分類。
3. 如果該菌株符合 WHO 高優先警戒類別（例如 carbapenem-resistant Acinetobacter 或 ESBL-producing Enterobacteriaceae），請標記為高通報風險。
4. 不是只列出內容，而是說明：
   - 哪些基因的組合意義重大？
   - 是否可能出現臨床治療無效？
   - 有無感染控制或公共衛生上的警訊？
5. 若樣本中出現極罕見的 AMR 組合或交叉抗藥性，請額外說明為何值得注意。

⚠️ 你的任務不是重複 CSV，而是提出臨床觀察與建議。

CSV 原始摘要如下：
{csv_content}
"""
}

# 預處理檔案（支援 tar、gz、csv）
from biom import load_table  # ✅ 新增
from io import BytesIO       # ✅ 用於處理 in-memory 檔案物件

def preprocess_uploaded_files(files):
    contents = {}
    for file in files:
        filename = file.name
        try:
            if filename.endswith(".tar") or filename.endswith(".tar.gz"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tar_path = os.path.join(tmpdir, filename)
                    with open(tar_path, "wb") as f:
                        f.write(file.read())
                    with tarfile.open(tar_path, "r:*") as tar:
                        tar.extractall(path=tmpdir)
                        for member in tar.getmembers():
                            if member.isfile() and member.name.endswith(".csv"):
                                csv_path = os.path.join(tmpdir, member.name)
                                df = pd.read_csv(csv_path)
                                contents[member.name] = df.head(20).to_csv(index=False)

            elif filename.endswith(".gz"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                    with gzip.open(file, "rb") as gz_file:
                        shutil.copyfileobj(gz_file, tmp_csv)
                    df = pd.read_csv(tmp_csv.name)
                    contents[filename[:-3]] = df.head(20).to_csv(index=False)

            elif filename.endswith(".biom"):
                biom_bytes = BytesIO(file.read())  # 將上傳的檔案轉為 in-memory stream
                table = load_table(biom_bytes)
                df = pd.DataFrame(
                    table.matrix_data.toarray(),
                    index=table.ids(axis='observation'),
                    columns=table.ids(axis='sample')
                )
                contents[filename] = df.head(20).to_csv(index=True)

            else:
                df = pd.read_csv(file)
                contents[filename] = df.head(20).to_csv(index=False)

        except Exception as e:
            contents[filename] = f"❌ 處理失敗: {e}"

    return contents


def check_filename_matches(expected_label, actual_filename):
    expected_keywords = expected_label.lower().split()
    filename_lower = actual_filename.lower()
    return all(keyword in filename_lower for keyword in expected_keywords)


def render_mode_card(icon, title, desc, key):
    selected = st.session_state.get("selected_mode") == title
    border = "4px solid #219ebc" if selected else "2px solid #ccc"
    shadow = "0 0 15px #219ebc" if selected else "none"
    bg = "#f0faff" if selected else "#ffffff"
    text_color = "#003049"

    with st.container():
        st.markdown(f"""
        <style>
        div#{key}_card {{
            background-color: {bg};
            color: {text_color};
            border-radius: 12px;
            border: {border};
            box-shadow: {shadow};
            padding: 1.5rem;
            height: 300px;
            text-align: center;
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        div#{key}_card:hover {{
            transform: scale(1.03);
            box-shadow: 0 0 20px #219ebc;
        }}
        div[data-testid="stButton"] > button#{key}_btn {{
            background-color: #219ebc;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            height: 40px;
            padding: 0 1.2rem;
        }}
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div id="{key}_card">
            <div style="font-size: 1.8rem;">{icon}</div>
            <div style="font-size: 1.6rem; font-weight: bold;">{title}</div>
            <div style="font-size: 1.3rem; color: #444;">{desc}</div>
            <div style="margin-top: 10px;">
        """, unsafe_allow_html=True)

        if st.button("選擇", key=f"{key}_btn"):
            st.session_state.selected_mode = title  # ❗不用 rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)
            
def select_mode(title):
    st.session_state.selected_mode = title

def render_mode_card(icon, title, desc, key):
    selected = st.session_state.get("selected_mode") == title
    bg = "#219ebc" if selected else "#ffffff"
    text_color = "#ffffff" if selected else "#003049"
    desc_color = "#e0f7ff" if selected else "#333333"
    border = "4px solid #219ebc" if selected else "2px solid #ccc"
    shadow = "0 0 25px #219ebc" if selected else "none"

    st.markdown(f"""
    <style>
    div#{key}_card {{
        background-color: {bg};
        color: {text_color};
        border: {border};
        border-radius: 16px;
        box-shadow: {shadow};
        padding: 1.5rem;
        height: 300px;
        text-align: center;
        transition: all 0.25s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    div#{key}_card:hover {{
        transform: scale(1.03);
        box-shadow: 0 0 30px #219ebc;
    }}
    div[data-testid="stButton"] > button#{key}_btn {{
        background-color: white;
        color: #219ebc;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        height: 40px;
        padding: 0 1.5rem;
        transition: background-color 0.2s;
    }}
    div[data-testid="stButton"] > button#{key}_btn:hover {{
        background-color: #d0f0ff;
        cursor: pointer;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div id="{key}_card">
        <header style="font-size: 2rem;">{icon}</header>
        <div style="font-size: 1.5rem; font-weight: bold;">{title}</div>
        <div style="font-size: 1.1rem; color: {desc_color}; margin-top: 0.5rem;">{desc}</div>
        <div style="margin-top: auto;">
    """, unsafe_allow_html=True)

    # ✅ 正確更新狀態並立刻影響 UI
    if st.button("選擇", key=f"{key}_btn", on_click=select_mode, args=(title,)):
        pass

    st.markdown("</div></div>", unsafe_allow_html=True)

def main():
    st.title("🧬 Gemini IDSEQ 分析儀表板")

    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = None
    
    st.markdown("""
    <div class="banner-text" style="background-color:#219ebc;color:white;text-align:center;
        padding:10px;border-radius:6px;margin:10px 0;font-weight:bold;font-size:16px;">
    請選擇分析主題
    </div>
    """, unsafe_allow_html=True)
    
    card_labels = list(TEMPLATE_MAP.keys())
    card_icons = ["🧬", "🧬", "🧬"]
    card_descs = ["微生物基因組分析", "病毒共識基因組比對", "抗藥性基因風險分析"]
    
    cols = st.columns(len(card_labels))
    for i, (icon, label, desc) in enumerate(zip(card_icons, card_labels, card_descs)):
        with cols[i]:
            render_mode_card(icon, label, desc, key=f"mode_{i}")


    # ===== 主流程（依主題呈現對應的上傳欄位） =====
    if st.session_state.selected_mode:
        mode = st.session_state.selected_mode

        mode_file_fields = {
            "Metagenomics": [
                "Heatmap", "Sample Metadata", "Samples Overview",
                "Sample Taxon Report", "Combined Sample Taxon Results",
                "Contig Summary Reports", "Host Genes Count", "Combined Microbiome File"
            ],
            "Consensus Genome": [
                "Consensus Genome", "Sample Metadata", "Consensus Genome Overview", "Intermediate Output Files"
            ],
            "Antimicrobial Resistance": [
                "Antimicrobial Assistance Results", "Combined AMR Results", "Sample Metadata"
            ]
        }

        uploaded_files_dict = {}

        for label in mode_file_fields[mode]:
            st.markdown(f"##### 📄 上傳：{label}")
            uploaded_file = st.file_uploader(
                f"📄 上傳：{label}",
                type=["csv", "gz", "tar", "biom"],
                key=f"uploader_{label}"
            )
            
            if uploaded_file is not None:
                # 只在第一次上傳時儲存進 session_state，避免重複執行
                if f"uploaded_{label}" not in st.session_state:
                    if check_filename_matches(label, uploaded_file.name):
                        st.session_state[f"uploaded_{label}"] = uploaded_file
                    else:
                        st.error(f"❌ 檔案名稱「{uploaded_file.name}」與預期欄位「{label}」不符")
            
            # 顯示已成功的檔案
            if f"uploaded_{label}" in st.session_state:
                uploaded_files_dict[label] = st.session_state[f"uploaded_{label}"]
                st.success(f"✅ 已上傳：{st.session_state[f'uploaded_{label}'].name}")
            if f"uploaded_{label}" in st.session_state:
                uploaded_files_dict[label] = st.session_state[f"uploaded_{label}"]
                st.success(f"✅ 已上傳 {st.session_state[f'uploaded_{label}'].name}")

        if uploaded_files_dict:
            st.success(f"✅ 已上傳 {len(uploaded_files_dict)} 個檔案")
            for name in uploaded_files_dict:
                st.write(f"- {name}")

        if st.button("🚀 Gemini + RAG 開始分析"):
            if not uploaded_files_dict:
                st.warning("請上傳檔案")
                return

            file_contents = preprocess_uploaded_files(uploaded_files_dict.values())
            prompt = generate_llm_prompt(mode, file_contents)

            with st.spinner("Gemini 分析中..."):
                try:
                    response = chat.send_message(prompt)
                    st.subheader("📄 分析結果")
                    st.markdown(f"""
                    <div style="background-color:#f7f9fc;padding:1.2rem 1.5rem;border-radius:12px;
                                border-left:6px solid #1f77b4;margin-bottom:1rem;">
                        <h4 style="margin-bottom:0.8rem;">📄 Gemini 分析結果</h4>
                        <pre style="white-space:pre-wrap;font-size:0.92rem;font-family:inherit;">
{response.text}</pre></div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Gemini 分析失敗：{e}")

            if st.checkbox("📊 顯示原始 CSV 摘要"):
                for name, content in file_contents.items():
                    st.write(f"📄 {name}")
                    st.code(content, language="csv")

if __name__ == "__main__":
    main()


