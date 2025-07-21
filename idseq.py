# ✅ 整合 FAISS RAG 到現有 app.py
import streamlit as st
import os 
import zipfile
import tarfile
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

from langchain_community.embeddings import HuggingFaceEmbeddings


INDEX_FILE_PATH = "microbio_faiss_index1.zip"
PDF_PATH = "C:\\Users\\User\\Downloads\\ilovepdf_merged.pdf"


def extract_index_archive(archive_path, extract_to="temp_faiss_index"):
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    return extract_to

def find_faiss_index_folder(base_path):
    for root, dirs, files in os.walk(base_path):
        if "index.faiss" in files and "index.pkl" in files:
            return root
    return None

def load_or_create_faiss():
    embedding = HuggingFaceEmbeddings()

    if INDEX_FILE_PATH.endswith((".zip", ".tar.gz", ".tgz")):
        extracted_dir = extract_index_archive(INDEX_FILE_PATH)
        index_dir = find_faiss_index_folder(extracted_dir)
    else:
        index_dir = INDEX_FILE_PATH

    if index_dir and os.path.exists(os.path.join(index_dir, "index.faiss")):
        return FAISS.load_local(index_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    else:
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"找不到 PDF：{PDF_PATH}")

        loader = PyMuPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split_documents(docs)
        texts = [chunk.page_content for chunk in chunks]
        vector_store = FAISS.from_texts(texts, embedding)
        vector_store.save_local(index_dir)
        return vector_store

    
# ✅ 初始化 Gemini
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY","")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro")
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

請你根據上述所有可能出現的資訊回答以下你在現有資訊中有能力回答的問題，綜合分析樣本中：

1. 樣本中是否出現臨床高風險或 WHO 高警戒的病原體？若有，出現在哪些樣本？

2. 有哪些微生物在整體中為高豐度或高頻率出現，具有流行病學或臨床意義？

3. 是否有僅在特定樣本中出現的稀有菌種？這些菌是否可能代表特定感染源（如環境或醫療相關）？

4. 各樣本間的菌群結構是否存在顯著差異？是否可形成群聚或代表特定病程？

5. 請列出高豐度菌種與其對應的樣本，並指出這是否與臨床診斷吻合。

6. 是否有樣本出現高度相似的菌群組成，可能提示交叉感染或院內感染？

7. 不同樣本來源（如 BAL、stool）是否對應到特定菌群組成？

8. 是否有特定菌種在不同時間或樣本中反覆出現，提示潛在的持續感染或環境污染源？

9. 樣本的採集來源或地理資訊是否與特定菌群分佈有關？

10. 有無樣本因品質控制不佳（如低非宿主 reads、總 reads 過低）而不適合進行分析？應排除哪些樣本？

11. 是否存在人類 reads 過高導致背景干擾的情況？

12. 哪些菌種的比對覆蓋率與深度最佳，可信度高？是否有菌僅部分基因片段比對成功？

13. 有哪些比對結果品質低下？是否影響菌種的診斷可靠性？

14. 根據宿主基因表現量資料，是否可看出發炎、免疫反應或感染階段變化？

15. 是否可推測疾病嚴重度（如病毒感染風險）或與特定臨床狀況（如 ICU 住院）有關的菌群變化？


📌 分析結果請條列整理，重點聚焦於臨床判讀與樣本風險評估，不必逐項報表說明。

CSV 原始摘要如下：
{csv_content}
""",
    "Consensus Genome": """
你是病毒基因組分析專家，請根據 Consensus Genome 比對與 QC 統計，做出專業見解：

1. 此樣本中的病毒是否有完整的共識基因組？覆蓋率與深度是否足夠進行變異分析？
2. 有哪些與參考病毒株不同的 SNP 或 INDEL？這些突變可能在哪些基因區？
3. 此樣本是否屬於某已知病毒株的 lineage？是否可能為新變異株？
4. 觀察到的突變是否與已知免疫逃逸、傳染性增加或藥物抗性有關？
5. 是否能根據共識序列與 metadata 判斷可能的病毒傳播鏈？樣本是否與其他樣本存在親緣關係？
6. 是否存在同一地點/同一病房中採集的樣本出現相似的病毒突變組合？是否構成群聚？
7. 是否有出現與已知高風險變異株（如 XBB.1.5、BA.2.86）相似的突變？
8. 此樣本的共識基因組是否足夠完整進行公共衛生通報或系統發佈（如 GISAID）？
9. 有哪些樣本應重新測序？例如低覆蓋、過多 N base、只覆蓋片段等情況？
10. 是否需要進一步確認特定突變的準確性？例如在低深度或 low-complexity 區域？

你的任務是根據 CSV 整合出臨床參考價值的分析，不是單純列出統計值。

CSV 原始摘要如下：
{csv_content}
""",  # 省略內容，保留原本
    "Antimicrobial Resistance": """
你是一位臨床感染科與抗藥性基因體分析專家，請根據下列 CSV（Combined AMR Results）進行風險導向的分析與臨床見解判斷：

1. 各樣本中偵測到哪些抗藥性基因？請列出其對應的抗生素類別與作用機制（如 β-lactamase, efflux pump, target modification）。
2. 哪些基因對應到 WHO 公告的「極高優先級」耐藥菌（如 CRE, ESBL, MRSA, VRE）？
3. 是否偵測到高風險的多重抗藥性基因組合？（如：carbapenemase + porin loss + efflux pump），請標記為高通報風險。
4. 根據每個樣本的 AMR 基因型，推薦可能仍有效的抗生素類別（如 polymyxin, tigecycline）？是否有任何一線藥物完全失效？
5. 是否有樣本屬於 MDR（多重抗藥性）或 XDR（廣泛抗藥性）定義？請依據 CDC/ECDC 分類標準判斷。
6. 根據藥物建議表，是否存在樣本無任何建議藥物？這些樣本可能需進一步送交培養藥敏試驗？
7. 是否有某些 AMR 基因與特定菌種高度共現？例如 NDM 與 Klebsiella, ermB 與 Streptococcus？
8. 是否有基因提示消毒劑抗性？（如 qacE, mdfA, tolC 等）影響感染控制措施？
9. 是否偵測到質體攜帶的 AMR 基因？可能具有高水平傳播風險？
10. 是否出現同一病房（或地點）中不同樣本帶有相似的 AMR 基因譜？是否可能為群聚感染？
11. 是否觀察到某些時間點 AMR 基因激增？是否暗示抗生素壓力下的選擇性繁殖？
12. 有沒有樣本來自社區但出現院內常見 AMR 樣式（如 ESBL-producing E. coli）？可能為社區擴散的早期警訊？



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
                        csv_files = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".csv")]
                        for member in csv_files:
                            csv_path = os.path.join(tmpdir, member.name)
                            df = pd.read_csv(csv_path)
                            if len(csv_files) > 1:
                                contents[member.name] = df.head(20).to_csv(index=True)
                            else:
                                contents[member.name] = df.to_csv(index=False)

            elif filename.endswith(".zip"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, filename)
                    with open(zip_path, "wb") as f:
                        f.write(file.read())
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                        csv_members = [m for m in zip_ref.namelist() if m.endswith(".csv")]
                        for member in csv_members:
                            member_path = os.path.join(tmpdir, member)
                            if os.path.isfile(member_path):
                                df = pd.read_csv(member_path)
                                if len(csv_members) > 1:
                                    contents[member] = df.head(20).to_csv(index=True)
                                else:
                                    contents[member] = df.to_csv(index=False)

            elif filename.endswith(".gz"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                    with gzip.open(file, "rb") as gz_file:
                        shutil.copyfileobj(gz_file, tmp_csv)
                    df = pd.read_csv(tmp_csv.name)
                    contents[filename[:-3]] = df.to_csv(index=False)

            elif filename.endswith(".biom"):
                biom_bytes = BytesIO(file.read())
                table = load_table(biom_bytes)
                df = pd.DataFrame(
                    table.matrix_data.toarray(),
                    index=table.ids(axis='observation'),
                    columns=table.ids(axis='sample')
                )
                contents[filename] = df.to_csv(index=True)

            else:
                df = pd.read_csv(file)
                contents[filename] = df.to_csv(index=False)

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
    # 若切換主題，重置已上傳檔案
    if st.session_state.get("selected_mode") != title:
        st.session_state.uploaded_files_dict = {}
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
    st.set_page_config(page_title="Gemini CSV 分析", layout="wide")
    st.title("🧬 Gemini IDSEQ 分析儀表板 (建議樣本數量至多十筆)") 

    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = None
    
    if "uploaded_files_dict" not in st.session_state:
        st.session_state.uploaded_files_dict = {}

    st.markdown("""
    <div class="banner-text" style="background-color:#219ebc;color:white;text-align:center;
        padding:10px;border-radius:6px;margin:10px 0;font-weight:bold;font-size:16px;">
    請選擇分析主題
    </div>
    """, unsafe_allow_html=True)

    card_labels = list(TEMPLATE_MAP.keys())
    card_icons = ["🧬", "🧬", "🧬"]
    card_descs = ["微生物基因組分析",
                  "病毒共識基因組比對", 
                  "抗藥性基因風險分析"]
    cols = st.columns(len(card_labels))
    for i, (icon, label, desc) in enumerate(zip(card_icons, card_labels, card_descs)):
        with cols[i]:
            render_mode_card(icon, label, desc, key=f"mode_{i}")

    if st.session_state.selected_mode:
        mode = st.session_state.selected_mode

        existing_keys = list(st.session_state.uploaded_files_dict.keys())
        for label in existing_keys:
            key = f"uploader_{mode}_{label}"
            if st.session_state.get(key) is None:
                del st.session_state.uploaded_files_dict[label]

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

        st.markdown("## 📂 上傳檔案")

        for label in mode_file_fields[mode]:
            uploaded_file = st.file_uploader(
                f"##### 📄 上傳：{label}",
                type=["csv", "gz", "tar", "biom", "zip"],
                key=f"uploader_{mode}_{label}"
            )
            if uploaded_file is not None:
                if check_filename_matches(label, uploaded_file.name):
                    st.session_state.uploaded_files_dict[label] = uploaded_file
                else:
                    st.error(f"❌ 檔案名稱「{uploaded_file.name}」與預期欄位「{label}」不符")

        if st.session_state.uploaded_files_dict:
            st.success(f"✅ 已上傳 {len(st.session_state.uploaded_files_dict)} 個檔案")
            for name in st.session_state.uploaded_files_dict:
                st.write(f"- {name}")

        # ✅ 按下按鈕才進行 Gemini 分析
        if st.button("🚀 Gemini + RAG 開始分析"):
            uploaded_files_dict = st.session_state.uploaded_files_dict
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

            if st.button("📊 清空分析結果"):
                file_contents = preprocess_uploaded_files(uploaded_files_dict.values())
                prompt = generate_llm_prompt(mode, file_contents)
            
                with st.spinner("Gemini 重新分析中..."):
                    try:
                        response = chat.send_message(prompt)
                        st.subheader("📄 分析結果（重新）")
                        st.markdown(f"""
                        <div style="background-color:#f7f9fc;padding:1.2rem 1.5rem;border-radius:12px;
                                    border-left:6px solid #1f77b4;margin-bottom:1rem;">
                            <h4 style="margin-bottom:0.8rem;">📄 Gemini 分析結果</h4>
                            <pre style="white-space:pre-wrap;font-size:0.92rem;font-family:inherit;">
            {response.text}</pre></div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Gemini 分析失敗：{e}")
            
                # 顯示摘要
                st.subheader("📊 原始 CSV 摘要")
                for name, content in file_contents.items():
                    st.write(f"📄 {name}")
                    st.code(content, language="csv")
                    
if __name__ == "__main__":
    main()
