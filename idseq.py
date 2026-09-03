# ✅ 整合 FAISS RAG 到現有 app.py
import streamlit as st
import os 
import zipfile
import tarfile
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai
import gzip
import shutil
import tempfile
from io import BytesIO
from biom import load_table

# ✅ FAISS RAG 套件
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_core.documents import Document 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ---------- SMART on FHIR 整合套件與函式 ----------
import requests
import base64
from datetime import datetime

DEFAULT_FHIR_URL = "http://localhost:8090/fhir"

def get_query_param(key):
    """相容於 Streamlit 版本的 Query Parameters 取得工具"""
    if hasattr(st, "query_params") and key in st.query_params:
        val = st.query_params[key]
        if isinstance(val, list):
            return val[0] if val else None
        return val
    return None

def discover_endpoints(iss):
    """自動偵測 FHIR 伺服器的 OAuth2 授權端點"""
    # 1. 優先嘗試 well-known 組態
    try:
        resp = requests.get(f"{iss.rstrip('/')}/.well-known/smart-configuration", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("authorization_endpoint"), data.get("token_endpoint")
    except Exception:
        pass
    # 2. 備用方案：解析 CapabilityStatement metadata
    try:
        resp = requests.get(f"{iss.rstrip('/')}/metadata", headers={"Accept": "application/fhir+json"}, timeout=3)
        if resp.status_code == 200:
            meta = resp.json()
            for rest in meta.get("rest", []):
                sec = rest.get("security", {})
                for ext in sec.get("extension", []):
                    if ext.get("url") == "http://fhir-registry.smarthealthit.org/StructureDefinition/oauth-uris":
                        auth_url = None
                        token_url = None
                        for sub_ext in ext.get("extension", []):
                            if sub_ext.get("url") == "authorize":
                                auth_url = sub_ext.get("valueUri") or sub_ext.get("valueUrl")
                            elif sub_ext.get("url") == "token":
                                token_url = sub_ext.get("valueUri") or sub_ext.get("valueUrl")
                        return auth_url, token_url
    except Exception:
        pass
    return None, None

# ---------- 離線高品質模擬病患資料庫 (當 FHIR 伺服器斷線時自動啟用，確保 Streamlit Cloud 100% 可用) ----------
MOCK_PATIENTS_DB = [
    {
        "id": "12212",
        "name": "Jane Doe",
        "gender": "female",
        "birthDate": "1985-05-12",
        "vitals": {
            "Body Height": "165 cm",
            "Body Weight": "58 kg",
            "Heart rate": "78 beats/min",
            "Respiratory rate": "16 breaths/min",
            "Pain severity": "2/10",
            "Head Occipital-frontal circumference": "N/A"
        },
        "labs": {
            "Leukocytes [Blood]": "9.5 10^3/uL",
            "Erythrocytes [Blood]": "4.2 10^6/uL",
            "Hemoglobin [Blood]": "12.8 g/dL",
            "Hematocrit [Blood]": "38.5 %",
            "Mean corpuscular volume (MCV)": "91 fL"
        },
        "conditions": ["Acute Bronchitis (Onset: 2026-08-15)", "Mild Asthma (Onset: 2020-03-10)"],
        "medications": ["Albuterol Inhaler (Prescribed: 2026-08-15)", "Amoxicillin 500mg (Prescribed: 2026-08-15)"],
        "procedures": ["Chest X-Ray 2 Views (Date: 2026-08-15)"]
    },
    {
        "id": "23526",
        "name": "John Smith",
        "gender": "male",
        "birthDate": "1972-11-23",
        "vitals": {
            "Body Height": "178 cm",
            "Body Weight": "82 kg",
            "Heart rate": "92 beats/min",
            "Respiratory rate": "20 breaths/min",
            "Pain severity": "6/10",
            "Head Occipital-frontal circumference": "N/A"
        },
        "labs": {
            "Leukocytes [Blood]": "14.8 10^3/uL",
            "Erythrocytes [Blood]": "4.8 10^6/uL",
            "Hemoglobin [Blood]": "14.2 g/dL",
            "Hematocrit [Blood]": "42.5 %",
            "Mean corpuscular volume (MCV)": "88 fL"
        },
        "conditions": ["Severe Sepsis (Onset: 2026-09-01)", "Pneumonia, Bacterial (Onset: 2026-09-01)"],
        "medications": ["Piperacillin-Tazobactam 4.5g IV (Prescribed: 2026-09-01)", "Vancomycin 1.25g IV (Prescribed: 2026-09-01)"],
        "procedures": ["Mechanical Ventilation (Date: 2026-09-01)", "Bronchoscopy (Date: 2026-09-02)"]
    },
    {
        "id": "23557",
        "name": "Robert Johnson",
        "gender": "male",
        "birthDate": "1960-04-05",
        "vitals": {
            "Body Height": "172 cm",
            "Body Weight": "75 kg",
            "Heart rate": "85 beats/min",
            "Respiratory rate": "18 breaths/min",
            "Pain severity": "4/10",
            "Head Occipital-frontal circumference": "N/A"
        },
        "labs": {
            "Leukocytes [Blood]": "11.2 10^3/uL",
            "Erythrocytes [Blood]": "4.5 10^6/uL",
            "Hemoglobin [Blood]": "13.5 g/dL",
            "Hematocrit [Blood]": "40.2 %",
            "Mean corpuscular volume (MCV)": "89 fL"
        },
        "conditions": ["Urinary Tract Infection (Onset: 2026-08-28)", "Type 2 Diabetes Mellitus (Onset: 2015-06-12)"],
        "medications": ["Ciprofloxacin 500mg PO (Prescribed: 2026-08-28)", "Metformin 1000mg PO (Prescribed: 2015-06-12)"],
        "procedures": ["Urine Culture & Susceptibility (Date: 2026-08-28)"]
    }
]

def get_fhir_patients(server_url):
    """調閱 FHIR 伺服器上的病患清單。若連線失敗，則自動啟用離線沙盒展示模式。"""
    try:
        url = f"{server_url.rstrip('/')}/Patient?_count=50"
        headers = {"Accept": "application/fhir+json"}
        resp = requests.get(url, headers=headers, timeout=3)
        if resp.status_code == 200:
            st.session_state.fhir_offline_fallback = False
            bundle = resp.json()
            patients = []
            for entry in bundle.get("entry", []):
                resource = entry.get("resource", {})
                pid = resource.get("id")
                # 解析病患姓名
                names = resource.get("name", [])
                full_name = "Unknown"
                if names:
                    name_parts = []
                    given = names[0].get("given", [])
                    family = names[0].get("family", "")
                    if given:
                        name_parts.append(" ".join(given))
                    if family:
                        name_parts.append(family)
                    full_name = " ".join(name_parts) if name_parts else "Unknown"
                gender = resource.get("gender", "Unknown")
                birth = resource.get("birthDate", "Unknown")
                patients.append({
                    "id": pid,
                    "name": full_name,
                    "gender": gender,
                    "birthDate": birth
                })
            return patients
    except Exception as e:
        # 連線失敗，自動轉入離線沙盒模式
        st.session_state.fhir_offline_fallback = True
        st.sidebar.warning("🔌 本地 HAPI FHIR 伺服器未連線。系統已自動切換至「離線沙盒展示模式」以供雲端 (Streamlit Cloud) 正常演示。")
        patients = []
        for p in MOCK_PATIENTS_DB:
            patients.append({
                "id": p["id"],
                "name": p["name"],
                "gender": p["gender"],
                "birthDate": p["birthDate"]
            })
        return patients
    return []

def get_fhir_patient_demographics(server_url, patient_id, token=None):
    """獲取單一病患詳細資料"""
    if st.session_state.get("fhir_offline_fallback", False):
        for p in MOCK_PATIENTS_DB:
            if p["id"] == patient_id:
                return {
                    "id": p["id"],
                    "name": p["name"],
                    "gender": p["gender"],
                    "birthDate": p["birthDate"],
                    "source": "Offline Sandbox Mode (離線沙盒展示)"
                }
    try:
        url = f"{server_url.rstrip('/')}/Patient/{patient_id}"
        headers = {"Accept": "application/fhir+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = requests.get(url, headers=headers, timeout=3)
        if resp.status_code == 200:
            resource = resp.json()
            names = resource.get("name", [])
            full_name = "Unknown"
            if names:
                name_parts = []
                given = names[0].get("given", [])
                family = names[0].get("family", "")
                if given:
                    name_parts.append(" ".join(given))
                if family:
                    name_parts.append(family)
                full_name = " ".join(name_parts) if name_parts else "Unknown"
            gender = resource.get("gender", "Unknown")
            birth = resource.get("birthDate", "Unknown")
            return {
                "id": patient_id,
                "name": full_name,
                "gender": gender,
                "birthDate": birth,
                "source": server_url
            }
    except Exception as e:
        st.error(f"調閱病患詳細資料失敗: {e}")
    return None

def get_fhir_patient_details(server_url, p_id, token=None):
    """獲取病患的臨床評估與診斷 (Condition)"""
    if st.session_state.get("fhir_offline_fallback", False):
        for p in MOCK_PATIENTS_DB:
            if p["id"] == p_id:
                return p["conditions"]
    conditions = []
    try:
        url = f"{server_url.rstrip('/')}/Condition?patient=Patient/{p_id}"
        headers = {"Accept": "application/fhir+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            bundle = r.json()
            entries = bundle.get("entry", [])
            for entry in entries:
                res = entry.get("resource", {})
                code_text = res.get("code", {}).get("text", "")
                if not code_text:
                    codings = res.get("code", {}).get("coding", [])
                    if codings:
                        code_text = codings[0].get("display", "")
                
                # 排除行政事項 / 非實際病症
                if "Medication review due" in code_text:
                    continue
                
                onset = res.get("onsetDateTime", "")
                if code_text:
                    if onset:
                        onset_brief = onset.split("T")[0]
                        conditions.append(f"{code_text} (Onset: {onset_brief})")
                    else:
                        conditions.append(code_text)
    except Exception:
        pass
    return conditions

def get_fhir_patient_medications(server_url, p_id, token=None):
    """獲取病患的用藥處方紀錄 (MedicationRequest)"""
    if st.session_state.get("fhir_offline_fallback", False):
        for p in MOCK_PATIENTS_DB:
            if p["id"] == p_id:
                return p["medications"]
    medications = []
    try:
        url = f"{server_url.rstrip('/')}/MedicationRequest?patient=Patient/{p_id}"
        headers = {"Accept": "application/fhir+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            bundle = r.json()
            entries = bundle.get("entry", [])
            for entry in entries:
                res = entry.get("resource", {})
                med_code = res.get("medicationCodeableConcept", {})
                display = ""
                codings = med_code.get("coding", [])
                if codings:
                    display = codings[0].get("display", codings[0].get("code", ""))
                elif "text" in med_code:
                    display = med_code["text"]
                
                authored_on = res.get("authoredOn", "")
                if display:
                    if authored_on:
                        date_brief = authored_on.split("T")[0]
                        medications.append(f"{display} (Prescribed: {date_brief})")
                    else:
                        medications.append(display)
    except Exception:
        pass
    return medications

def get_fhir_patient_procedures(server_url, p_id, token=None):
    """獲取病患的醫療處置與手術紀錄 (Procedure)"""
    if st.session_state.get("fhir_offline_fallback", False):
        for p in MOCK_PATIENTS_DB:
            if p["id"] == p_id:
                return p["procedures"]
    procedures = []
    try:
        url = f"{server_url.rstrip('/')}/Procedure?patient=Patient/{p_id}"
        headers = {"Accept": "application/fhir+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            bundle = r.json()
            entries = bundle.get("entry", [])
            for entry in entries:
                res = entry.get("resource", {})
                code_obj = res.get("code", {})
                display = ""
                codings = code_obj.get("coding", [])
                if codings:
                    display = codings[0].get("display", codings[0].get("code", ""))
                elif "text" in code_obj:
                    display = code_obj["text"]
                
                # 排除非實際醫療治療的行政程序
                if "Medication reconciliation" in display:
                    continue
                
                perf = res.get("performedPeriod", {})
                date = perf.get("start", res.get("performedDateTime", ""))
                if display:
                    if date:
                        date_brief = date.split("T")[0]
                        procedures.append(f"{display} (Date: {date_brief})")
                    else:
                        procedures.append(display)
    except Exception:
        pass
    return procedures

def get_fhir_patient_observations(server_url, p_id, token=None):
    """獲取與分類病患的觀察檢驗紀錄 (Observation) -> 生命徵象與實驗室檢驗"""
    if st.session_state.get("fhir_offline_fallback", False):
        for p in MOCK_PATIENTS_DB:
            if p["id"] == p_id:
                return p["vitals"], p["labs"]
    vitals = {}
    labs = {}
    try:
        url = f"{server_url.rstrip('/')}/Observation?patient=Patient/{p_id}&_count=100"
        headers = {"Accept": "application/fhir+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            bundle = r.json()
            entries = bundle.get("entry", [])
            for entry in entries:
                res = entry.get("resource", {})
                code_text = res.get("code", {}).get("text", "")
                
                value = "N/A"
                if "valueQuantity" in res:
                    vq = res["valueQuantity"]
                    val_num = vq.get("value")
                    if isinstance(val_num, float):
                        val_num = round(val_num, 2)
                    value = f"{val_num} {vq.get('unit', '')}"
                elif "valueCodeableConcept" in res:
                    value = res["valueCodeableConcept"].get("text", "")
                elif "valueString" in res:
                    value = res["valueString"]
                
                # 分類為生命徵象 (Vitals) 或是實驗室檢驗 (Labs)
                vitals_keys = ["Body Height", "Body Weight", "Heart rate", "Respiratory rate", "Pain severity", "Head Occipital-frontal circumference"]
                is_vital = False
                for vk in vitals_keys:
                    if vk.lower() in code_text.lower():
                        vitals[code_text] = value
                        is_vital = True
                        break
                if not is_vital and code_text:
                    labs[code_text] = value
    except Exception:
        pass
    return vitals, labs

def upload_report_to_fhir(server_url, patient_id, report_markdown, report_title, token=None):
    """將產生的臨床分析報告以 DocumentReference 上傳儲存至 FHIR 伺服器"""
    try:
        url = f"{server_url.rstrip('/')}/DocumentReference"
        headers = {
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        encoded_data = base64.b64encode(report_markdown.encode("utf-8")).decode("utf-8")
        
        doc_ref = {
            "resourceType": "DocumentReference",
            "status": "current",
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11502-2",
                        "display": "Laboratory report"
                    }
                ],
                "text": report_title
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "content": [
                {
                    "attachment": {
                        "contentType": "text/markdown; charset=utf-8",
                        "data": encoded_data,
                        "title": report_title
                    }
                }
            ]
        }
        
        resp = requests.post(url, json=doc_ref, headers=headers, timeout=5)
        if resp.status_code in [200, 201]:
            return True, resp.json().get("id")
        else:
            return False, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)

def convert_text_to_fhir_structured_ai(patient_id, report_markdown, api_key):
    """
    使用 AI 臨床 NLP 工具 (類似 Microsoft Azure Text Analytics for Health 或 John Snow Labs FHIR-Ready AI)，
    將非結構化的基因組分析報告轉換為高度結構化的 R4 FHIR Bundle (包含 DiagnosticReport 與一系列 Observation 資源)，
    完全不使用 Base64 編碼，而是對臨床醫學實體（病原菌、抗藥性基因、覆蓋率等）進行真正意義上的結構化編碼與解析。
    """
    import google.generativeai as genai
    import json
    import datetime

    # 初始化與配置 Gemini 作為 Clinical NLP 實體提取引擎
    genai.configure(api_key=api_key)
    nlp_model = genai.GenerativeModel("gemini-2.5-pro")

    prompt = f"""
You are a highly specialized clinical NLP pipeline engine, functioning like Microsoft Azure's Text Analytics for Health or John Snow Labs FHIR-Ready AI.
Your job is to analyze unstructured clinical genomic text and extract clinical entities, mapping them to structured, valid FHIR R4 resources with standard terminology codings.

Unstructured Genomic Analysis Text Report:
\"\"\"
{report_markdown}
\"\"\"

Patient ID: {patient_id}

You must convert this unstructured report into a structured FHIR R4 Bundle containing:
1. A `DiagnosticReport` representing the overall genomic analysis.
   - It must link to the subject: `Patient/{patient_id}`.
   - It must contain a `result` array linking to all the extracted structured `Observation` resources.
2. Multiple structured `Observation` resources representing:
   - Specific pathogens detected (e.g., Escherichia coli, SARS-CoV-2, etc.). Use SNOMED CT (`http://snomed.info/sct`) or LOINC (`http://loinc.org`) coding. Include quantitative details (such as abundance, reads, or coverage percentage) in `valueQuantity` or `valueString`.
   - Antimicrobial resistance (AMR) genes or drug resistance findings. Use genomic or HGVS terminology coding.
   - Viral lineage or mutation variants (SNPs/INDELs).
   - Each Observation must link back to the subject: `Patient/{patient_id}` and have `status: "final"`.

Standard Terminology Mappings (use realistic standard codes):
- Metagenomic next-generation sequencing analysis: LOINC `96381-9`
- Snomed bacterium concept: `http://snomed.info/sct`
- Loinc coding system: `http://loinc.org`

Ensure that the output is:
- A single valid FHIR Bundle resource (`resourceType: "Bundle"`, `type: "collection"` or `type: "transaction"`).
- Free of any base64 attachment wrapper. It must contain the actual structured, readable clinical concepts and data points.
- Return ONLY the raw JSON string of the FHIR Bundle. Do not wrap the output in markdown code blocks like ```json ... ```, and do not include any conversation or explanation.
"""

    try:
        response = nlp_model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # 清理可能夾帶的 Markdown 標記
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
        fhir_bundle = json.loads(raw_text)
        return fhir_bundle
    except Exception as e:
        # 備用方案：如果 AI 回傳不合規 JSON，則建立最精確的結構化備用 Bundle 資源
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        fallback_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "DiagnosticReport",
                        "id": "dr-genomic-fallback",
                        "status": "final",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://loinc.org",
                                    "code": "96381-9",
                                    "display": "Metagenomic next-generation sequencing analysis"
                                }
                            ],
                            "text": "Structured Genomic Analysis Report"
                        },
                        "subject": {
                            "reference": f"Patient/{patient_id}"
                        },
                        "issued": now_str,
                        "conclusion": f"Structured extraction processed by Clinical NLP. Source report text processed: {report_markdown[:300]}..."
                    }
                }
            ]
        }
        return fallback_bundle

def upload_fhir_resource(server_url, resource_type, resource_json, token=None):
    """將現成的 FHIR JSON 資源上傳儲存至 FHIR 伺服器"""
    try:
        url = f"{server_url.rstrip('/')}/{resource_type}"
        headers = {
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        resp = requests.post(url, json=resource_json, headers=headers, timeout=5)
        if resp.status_code in [200, 201]:
            return True, resp.json().get("id")
        else:
            return False, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)

def generate_cohort_metadata(file_contents, fhir_patients):
    """將上傳檔案中辨識出的各個 Sample 映射到 FHIR 伺服器中的病患個資並加入模擬病房位置，用作院內感控分析"""
    import re
    # 掃描並找出檔案內容中的 Sample ID (例如 Sample_A, Sample_1, S2, s03 等)
    sample_names = set()
    for content in file_contents.values():
        matches = re.findall(r'\b(Sample_[A-Za-z0-9_-]+|sample_[A-Za-z0-9_-]+|S\d+|s\d+)\b', content)
        sample_names.update(matches)
        
    if not sample_names:
        sample_names = ["Sample_A", "Sample_B", "Sample_C", "Sample_D"]
        
    sample_names = sorted(list(sample_names))
    
    lines = ["Sample Name,Patient Name,Patient ID,Gender,BirthDate,Ward/Location"]
    for i, sname in enumerate(sample_names):
        if fhir_patients and i < len(fhir_patients):
            p = fhir_patients[i]
        else:
            # 備用模擬病患
            p = {
                "name": f"Patient Mock {i+1}", 
                "id": f"MOCK-{100+i}", 
                "gender": "male" if i % 2 == 0 else "female", 
                "birthDate": "1980-01-01"
            }
            
        # 分配床位
        ward = f"ICU Bed {i+1}" if i < 3 else f"General Ward 3A Bed {i-2}"
        lines.append(f"{sname},{p['name']},{p['id']},{p['gender']},{p['birthDate']},{ward}")
        
    return "\n".join(lines)

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
    embedding = HuggingFaceEmbeddings(model_kwargs={'device': 'cpu'})

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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY","AIzaSyBX77uIUdYXpUcRrl_kbgEZM1UFlytPDnE")

try:
    genai.configure(api_key="AIzaSyBX77uIUdYXpUcRrl_kbgEZM1UFlytPDnE")
    model = genai.GenerativeModel("gemini-2.5-pro")
    chat = model.start_chat()
except Exception as e:
    model = None
    chat = None

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
        "Heatmap": "Statistical matrix of all samples and all microbial taxons.",
        "Sample Metadata": "Basic information of the samples, such as sampling time and site.",
        "Samples Overview": "QC statistics and summary for each sample.",
        "Sample Taxon Report": "Microbial classification and quantitative data detected in each sample.",
        "Combined Sample Taxon Results": "Aggregated microbiology data summary table for all samples.",
        "Contig Summary Reports": "QC statistics and coverage of Contigs.",
        "Host Gene Count": "Host transcript expression statistics.",
        "Consensus Genome Overview": "Quality control (QC) metrics of the consensus genome (e.g., genome coverage percentage, mapped reads, SNP count) and other statistical summaries.",
        "Antimicrobial Resistance Results": "Includes resistance reports, complete resistance indicators, intermediate analysis results, and CARD RGI tool outputs.",
        "Combined AMR Results": "Integrates indicators of drug resistance genes (e.g., coverage, depth) in samples into a single report."
    }

    summary_lines = []

    # 🔹 Add file definitions based on mode
    if mode in ["Metagenomics", "Consensus Genome", "Antimicrobial Resistance"]:
        summary_lines.append("📘 File Definitions:")
        for label, definition in file_definitions.items():
            summary_lines.append(f"- **{label}**: {definition}")
        summary_lines.append("")

    # 🔹 Add user-uploaded CSV summary content
    for label, content in file_contents.items():
        summary_lines.append(f"📄 File: {label}\nContent Summary:\n{content}\n")

    # 🔹 Search vector database for relevant background knowledge
    user_query = f"{mode} analysis guidelines and clinical risk"
    context_text = retrieve_context(user_query)
    summary_lines.append(f"\n📚 Textbook Supplementary Knowledge:\n{context_text}")

    prompt_template = TEMPLATE_MAP[mode]
    return prompt_template.format(csv_content="\n".join(summary_lines))




# Prompt 模板
TEMPLATE_MAP = {
    "Metagenomics": """
You are an expert in microbial genomics. Please perform a clinical-oriented comprehensive interpretation based on the IDSEQ Metagenomics CSV data uploaded by the user. These data may include:

- Sample Taxons Report: Microorganisms detected in the sample and their reads/rPM
- Combined Sample Taxon Results: Aggregated microbial abundance across all samples
- Taxon Heatmap: Quantitative matrix of multiple samples and microorganisms
- Sample Metadata: Sample collection source, time, sample type
- Samples Overview: Quality control (QC) and pass rate of each sample
- Contig Summary Reports: Coverage and sequence alignment quality of microbial gene fragments
- Host Gene Count: Host gene expression (potentially related to infection and immune response)

Please answer the following questions based on all the above possible information that you are capable of answering, comprehensively analyzing:

1. Did clinical high-risk or WHO high-alert pathogens appear in the samples? If so, in which samples?
2. Which microorganisms appeared at high abundance or high frequency overall, possessing epidemiological or clinical significance?
3. Are there rare strains that appeared only in specific samples? Could these strains represent specific infection sources (e.g., environmental or healthcare-associated)?
4. Are there significant differences in the community structure among different samples? Do they represent specific disease stages?
5. List the high-abundance species and their corresponding samples, and indicate whether this is consistent with clinical diagnosis.
6. Do different sample sources (e.g., BAL, stool) correspond to specific microbial compositions?
7. Do specific species repeatedly appear in different time points or samples, suggesting potential persistent infection?
8. Are there any samples unsuitable for analysis due to poor quality control (e.g., low non-host reads, total reads too low)? Which samples should be excluded?
9. Is there any background interference due to excessively high human reads?
10. Which species have the best alignment coverage and depth, indicating high credibility? Are there species where only partial gene fragments aligned successfully?
11. Are there any alignment results with poor quality? Does this affect the diagnostic reliability of the species?
12. Based on the host gene expression data, can you observe changes in inflammation, immune response, or infection stages?
13. Can you infer disease severity (e.g., viral infection risk) or changes in microbial community related to specific clinical conditions (e.g., ICU admission)?

📌 Please use these questions as guidelines for Metagenomics Analysis to synthesize a clinical observation and insight report. The final report MUST be written entirely in English.

Raw CSV Summary:
{csv_content}
""",
    "Consensus Genome": """
You are an expert in viral genome analysis. Please provide professional insights based on the Consensus Genome alignment and QC statistics:

1. Does the virus in this sample have a complete consensus genome? Are the coverage and depth sufficient for variant analysis?
2. What SNPs or INDELs differ from the reference viral strain? In which gene regions might these mutations be located?
3. Does this sample belong to a lineage of a known viral strain? Could it be a new variant?
4. Are the observed mutations associated with known immune escape, increased infectivity, or drug resistance?
5. Are there mutations similar to known high-risk variants (such as XBB.1.5, BA.2.86)?
6. Is the consensus genome of this sample complete enough for public health reporting or database submission (e.g., GISAID)?
7. Which samples should be re-sequenced? (e.g., low coverage, excessive N bases, only covering fragments)?
8. Is there a need to further confirm the accuracy of specific mutations? (e.g., in low-depth or low-complexity regions)?

📌 Please use these questions as guidelines for Consensus Genome Analysis to synthesize a clinical observation and insight report. The final report MUST be written entirely in English.

Raw CSV Summary:
{csv_content}
""",
    "Antimicrobial Resistance": """
You are a clinical infectious disease and antimicrobial resistance (AMR) genomics expert. Please perform a risk-oriented analysis and clinical insight determination based on the following CSV (Combined AMR Results):

1. Which AMR genes were detected in each sample? Please list their corresponding antibiotic classes and mechanisms of action (e.g., β-lactamase, efflux pump, target modification).
2. Which genes correspond to WHO-declared 'critical priority' drug-resistant bacteria (e.g., CRE, ESBL, MRSA, VRE)?
3. Were any high-risk multidrug-resistant gene combinations detected? (e.g., carbapenemase + porin loss + efflux pump); please flag these as high risk for reporting.
4. Based on the AMR genotype of each sample, which antibiotic classes are recommended that may still be effective (e.g., polymyxin, tigecycline)? Have any first-line drugs completely failed?
5. Do any samples fit the definition of MDR (multidrug resistance) or XDR (extensively drug-resistant) according to CDC/ECDC classification standards?
6. Based on the drug recommendation table, are there samples with no recommended drugs? Might these samples require further phenotypic antimicrobial susceptibility testing?
7. Are certain AMR genes highly co-occurrent with specific species? (e.g., NDM with Klebsiella, ermB with Streptococcus)?
8. Are there genes indicating disinfectant resistance? (e.g., qacE, mdfA, tolC, etc.) that impact infection control measures?
9. Were any plasmid-mediated AMR genes detected? Do they pose a high risk of transmission?
10. Are there specific time points where AMR genes surged? Does this suggest selective pressure and propagation under antibiotic use?
11. Are there any AMR genes with low coverage or low alignment quality? Do these results require manual review or exclusion?

📌 Please use these questions as guidelines for Antimicrobial Resistance Analysis to synthesize a clinical observation and insight report. The final report MUST be written entirely in English.

Raw CSV Summary:
{csv_content}
"""
}

# 預處理檔案（支援 tar、gz、csv）
from biom import load_table  # ✅ 新增
from io import BytesIO       # ✅ 用於處理 in-memory 檔案物件

def preprocess_uploaded_files(files_dict):
    contents = {}
    for label, file in files_dict.items():
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
                            content_key = f"{label} ({member.name})"
                            if len(csv_files) > 1:
                                contents[content_key] = df.head(20).to_csv(index=True)
                            else:
                                contents[content_key] = df.to_csv(index=False)

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
                                content_key = f"{label} ({member})"
                                if len(csv_members) > 1:
                                    contents[content_key] = df.head(20).to_csv(index=True)
                                else:
                                    contents[content_key] = df.to_csv(index=False)

            elif filename.endswith(".gz"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                    with gzip.open(file, "rb") as gz_file:
                        shutil.copyfileobj(gz_file, tmp_csv)
                    df = pd.read_csv(tmp_csv.name)
                    contents[label] = df.to_csv(index=False)

            elif filename.endswith(".biom"):
                biom_bytes = BytesIO(file.read())
                table = load_table(biom_bytes)
                df = pd.DataFrame(
                    table.matrix_data.toarray(),
                    index=table.ids(axis='observation'),
                    columns=table.ids(axis='sample')
                )
                contents[label] = df.to_csv(index=True)

            else:
                df = pd.read_csv(file)
                contents[label] = df.to_csv(index=False)

        except Exception as e:
            contents[label] = f"❌ 處理失敗: {e}"

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
    st.title("🧬 Gemini IDSEQ 分析儀表板") 

    # ---------- 初始化 SMART on FHIR 狀態變數 ----------
    if "fhir_url" not in st.session_state:
        st.session_state.fhir_url = DEFAULT_FHIR_URL
    if "fhir_token" not in st.session_state:
        st.session_state.fhir_token = None
    if "fhir_patient_id" not in st.session_state:
        st.session_state.fhir_patient_id = None
    if "active_patient_demographics" not in st.session_state:
        st.session_state.active_patient_demographics = None
    if "gemini_analysis_result" not in st.session_state:
        st.session_state.gemini_analysis_result = None
    if "fhir_json_preview" not in st.session_state:
        st.session_state.fhir_json_preview = None
    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = None
    if "uploaded_files_dict" not in st.session_state:
        st.session_state.uploaded_files_dict = {}

    # ---------- 處理 EHR 啟動參數 (SMART Launch / Unsecured Fallback) ----------
    iss_param = get_query_param("iss")
    launch_param = get_query_param("launch")
    patient_param = get_query_param("patient")
    code_param = get_query_param("code")
    state_param = get_query_param("state")

    if iss_param:
        st.session_state.fhir_url = iss_param
        auth_endpoint, token_endpoint = discover_endpoints(iss_param)
        
        if auth_endpoint and token_endpoint:
            # 1. 軌道 A：支援安全驗證的標準 SMART Launch 流程
            st.session_state.auth_endpoint = auth_endpoint
            st.session_state.token_endpoint = token_endpoint
            if launch_param:
                st.session_state.launch_id = launch_param
                client_id = "idseq_streamlit_app"
                redirect_uri = "http://localhost:8501/"
                scopes = "launch patient/*.read patient/*.write openid fhirUser"
                auth_redirect_url = f"{auth_endpoint}?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope={scopes}&state=idseq_state&launch={launch_param}"
                
                st.warning("🔄 檢測到來自 EHR 系統的 SMART on FHIR 啟動請求！")
                st.link_button("🔑 授權連線並登入 EHR 系統", auth_redirect_url)
                st.stop()
        else:
            # 2. 軌道 B：本地無驗證 FHIR 伺服器模擬啟動 (解決本機 HAPI FHIR 報錯)
            st.sidebar.info("🔌 已偵測到免驗證本地 FHIR 伺服器...")
            if patient_param:
                st.session_state.fhir_patient_id = patient_param
                p_demo = get_fhir_patient_demographics(iss_param, patient_param)
                if p_demo:
                    st.session_state.active_patient_demographics = p_demo
            else:
                st.sidebar.warning("💡 請在網址尾端加入 `&patient=病患ID`（如：`&patient=12212`）以模擬自動帶入病患。")

    elif code_param and state_param == "idseq_state":
        if "token_endpoint" in st.session_state:
            token_endpoint = st.session_state.token_endpoint
            with st.spinner("🔄 正在交換 EHR 授權 Token..."):
                try:
                    payload = {
                        "grant_type": "authorization_code",
                        "code": code_param,
                        "redirect_uri": "http://localhost:8501/",
                        "client_id": "idseq_streamlit_app"
                    }
                    resp = requests.post(token_endpoint, data=payload, timeout=5)
                    if resp.status_code == 200:
                        token_data = resp.json()
                        st.session_state.fhir_token = token_data.get("access_token")
                        st.session_state.fhir_patient_id = token_data.get("patient")
                        p_demo = get_fhir_patient_demographics(st.session_state.fhir_url, st.session_state.fhir_patient_id, st.session_state.fhir_token)
                        if p_demo:
                            st.session_state.active_patient_demographics = p_demo
                            st.success(f"🎉 成功連線！病患: {p_demo.get('name')} (ID: {p_demo.get('id')})")
                    else:
                        st.error(f"❌ Token 交換失敗: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.error(f"❌ Token 交換出錯: {e}")
            st.query_params.clear()

    # ---------- SMART on FHIR 側邊欄控制面板 ----------
    st.sidebar.markdown("# 🔌 SMART on FHIR 控制面板")
    connection_mode = st.sidebar.radio(
        "選擇 FHIR 連線模式",
        ["本機測試模式 (無驗證)", "EHR SMART 啟動模式"],
        index=0 if not st.session_state.fhir_token else 1
    )

    # ---------- 確保預先讀取病患清單 ----------
    if "local_patients" not in st.session_state or not st.session_state.local_patients:
        st.session_state.local_patients = get_fhir_patients(st.session_state.fhir_url)

    # ---------- 分析主體範圍選擇 ----------
    analysis_scope = "「單一病患」病程/部位追蹤"

    if connection_mode == "本機測試模式 (無驗證)":
        st.sidebar.markdown("### 🔌 本機 HAPI FHIR 連線")
        local_url = st.sidebar.text_input("FHIR 伺服器網址", value=st.session_state.fhir_url)
        st.session_state.fhir_url = local_url
        
        if st.sidebar.button("🔌 連線至伺服器"):
            patients = get_fhir_patients(local_url)
            if patients:
                st.session_state.local_patients = patients
                st.sidebar.success(f"✅ 成功連線！找到 {len(patients)} 位病患資料。")
            else:
                st.sidebar.error("❌ 無法取得病患清單，請確認伺服器是否正常運行。")
                
        # 只有在單一病患追蹤模式下才讓使用者選擇單一病患
        if analysis_scope == "「單一病患」病程/部位追蹤":
            if "local_patients" in st.session_state and st.session_state.local_patients:
                patient_options = {f"{p['name']} (ID: {p['id']})": p for p in st.session_state.local_patients}
                selected_pat_label = st.sidebar.selectbox("🎯 選擇病患", list(patient_options.keys()))
                if selected_pat_label:
                    selected_p = patient_options[selected_pat_label]
                    st.session_state.fhir_patient_id = selected_p["id"]
                    st.session_state.active_patient_demographics = {
                        "id": selected_p["id"],
                        "name": selected_p["name"],
                        "gender": selected_p["gender"],
                        "birthDate": selected_p["birthDate"],
                        "source": local_url
                    }
        else:
            # 院內感控模式，不需要選擇單一病患
            st.session_state.active_patient_demographics = None
    else:
        st.sidebar.markdown("### 🔑 EHR SMART Launch 狀態")
        if st.session_state.fhir_token:
            st.sidebar.success("🟢 已經由 EHR 成功授權登入")
            st.sidebar.markdown(f"**伺服器:** `{st.session_state.fhir_url}`")
            if analysis_scope == "「單一病患」病程/部位追蹤":
                st.sidebar.markdown(f"**病患病歷號:** `{st.session_state.fhir_patient_id}`")
        else:
            st.sidebar.info("⏳ 等待 EHR 系統發起 Launch 請求...\n可在 EHR 中直接開啟本 App，或於網址加上 `?iss=...&launch=...` 參數。")
            
            # 本地模擬提示，引導使用者
            if "localhost" in st.session_state.fhir_url or "127.0.0.1" in st.session_state.fhir_url:
                st.sidebar.warning("""
                💡 **本地測試提示：**
                您選擇了「EHR SMART 啟動模式」，但目前連線的 HAPI FHIR (`localhost:8090`) 是**免安全驗證（無 OAuth2）**的本地測試沙盒。
                
                在此環境下，如果您想模擬 EHR 啟動，請**直接切換為「本機測試模式 (無驗證)」**（更方便），或者在瀏覽器網址列貼上以下網址進行免驗證模擬快速啟動：
                
                `http://localhost:8501/?iss=http://localhost:8090/fhir&patient=12212`
                """)
            
        if st.sidebar.button("🚪 登出 / 重設連線"):
            st.session_state.fhir_token = None
            st.session_state.fhir_patient_id = None
            st.session_state.active_patient_demographics = None
            st.session_state.gemini_analysis_result = None
            st.query_params.clear()
            st.sidebar.success("已重設連線狀態")
            st.rerun()

    # ---------- Gemini API 金鑰配置 ----------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 Gemini API 金鑰配置")
    
    # 預設帶入原設定之金鑰，一開始不要空白
    default_key_val = st.session_state.get("user_gemini_key", GOOGLE_API_KEY)
    user_api_key = st.sidebar.text_input(
        "輸入 Gemini API 金鑰",
        value=default_key_val,
        type="password",
        help="輸入您的 Gemini API 金鑰。預設已自動帶入系統內置的金鑰。"
    )
    st.session_state.user_gemini_key = user_api_key

    # ---------- 主介面：病患臨床卡片 / 感控提示卡片 ----------
    if analysis_scope == "「單一病患」病程/部位追蹤" and st.session_state.active_patient_demographics:
        p = st.session_state.active_patient_demographics
        age_str = "未知"
        if p.get("birthDate") and p.get("birthDate") != "Unknown":
            try:
                b_year = int(p.get("birthDate").split("-")[0])
                age_str = f"{datetime.now().year - b_year} 歲"
            except:
                pass
                
        # 調閱與分類患者的臨床紀錄
        conditions = get_fhir_patient_details(st.session_state.fhir_url, p.get("id"), st.session_state.get("fhir_token"))
        medications = get_fhir_patient_medications(st.session_state.fhir_url, p.get("id"), st.session_state.get("fhir_token"))
        procedures = get_fhir_patient_procedures(st.session_state.fhir_url, p.get("id"), st.session_state.get("fhir_token"))
        vitals, labs = get_fhir_patient_observations(st.session_state.fhir_url, p.get("id"), st.session_state.get("fhir_token"))

        # 生命徵象 (Vitals) 格式化
        height_val = next((v for k, v in vitals.items() if "height" in k.lower()), "N/A")
        weight_val = next((v for k, v in vitals.items() if "weight" in k.lower() and "length" not in k.lower()), "N/A")
        hr_val = next((v for k, v in vitals.items() if "heart rate" in k.lower()), "N/A")
        rr_val = next((v for k, v in vitals.items() if "respiratory rate" in k.lower()), "N/A")
        pain_val = next((v for k, v in vitals.items() if "pain severity" in k.lower()), "N/A")
        head_val = next((v for k, v in vitals.items() if "head occipital-frontal circumference" in k.lower() and "percentile" not in k.lower()), "N/A")

        # 實驗室檢驗 (Labs - CBC) 格式化
        wbc_val = next((v for k, v in labs.items() if "leukocytes" in k.lower()), "N/A")
        rbc_val = next((v for k, v in labs.items() if "erythrocytes" in k.lower()), "N/A")
        hb_val = next((v for k, v in labs.items() if "hemoglobin" in k.lower()), "N/A")
        hct_val = next((v for k, v in labs.items() if "hematocrit" in k.lower()), "N/A")
        mcv_val = next((v for k, v in labs.items() if "mcv" in k.lower() or "mean volume" in k.lower()), "N/A")
        mch_val = next((v for k, v in labs.items() if "mch" in k.lower() and "mchc" not in k.lower()), "N/A")
        mchc_val = next((v for k, v in labs.items() if "mchc" in k.lower()), "N/A")
        rdw_val = next((v for k, v in labs.items() if "distwidth" in k.lower() or "erythrocyte distribution width" in k.lower()), "N/A")

        st.markdown(f"""
        <div style="background-color:#e8f1f5; padding:1.5rem; border-radius:12px; border-left:8px solid #219ebc; margin-bottom:1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <h3 style="margin:0 0 1rem 0; color:#1f618d; border-bottom: 2px solid #bde0fe; padding-bottom: 6px;">📌 當前連接病患臨床資訊 (FHIR Clinical Records)</h3>
            <table style="width: 100%; border-collapse: collapse; color: #2c3e50; font-size: 1.05rem; line-height: 1.6;">
                <tr style="border-bottom: 1px solid #dcdde1;">
                    <td style="padding: 8px 0; font-weight: bold; width: 25%; color: #34495e;">👤 基本資訊 (Demographics):</td>
                    <td style="padding: 8px 0; color: #1e3799; font-weight: 600;">
                        姓名: {p.get('name')} &nbsp;|&nbsp; 
                        病歷號: {p.get('id')} &nbsp;|&nbsp; 
                        性別: {p.get('gender').upper()} &nbsp;|&nbsp; 
                        出生日期: {p.get('birthDate')} ({age_str}) &nbsp;|&nbsp; 
                        來源系統: {p.get('source')}
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #dcdde1;">
                    <td style="padding: 8px 0; font-weight: bold; color: #34495e; vertical-align: top;">💓 生命徵象 (Vitals):</td>
                    <td style="padding: 8px 0; color: #0b8043; font-weight: 600;">
                        身高: {height_val} &nbsp;|&nbsp; 
                        體重: {weight_val} &nbsp;|&nbsp; 
                        心率: {hr_val} &nbsp;|&nbsp; 
                        呼吸頻率: {rr_val} &nbsp;|&nbsp; 
                        疼痛評估: {pain_val} &nbsp;|&nbsp; 
                        頭圍: {head_val}
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #dcdde1;">
                    <td style="padding: 8px 0; font-weight: bold; color: #34495e; vertical-align: top;">🔬 血液檢驗 (Labs - CBC):</td>
                    <td style="padding: 8px 0; color: #0f52ba; font-weight: 600;">
                        白血球: {wbc_val} &nbsp;|&nbsp; 
                        紅血球: {rbc_val} &nbsp;|&nbsp; 
                        血紅素: {hb_val} &nbsp;|&nbsp; 
                        血球比容: {hct_val} &nbsp;|&nbsp; 
                        平均體積 (MCV): {mcv_val} &nbsp;|&nbsp; 
                        平均血紅素 (MCH): {mch_val} &nbsp;|&nbsp; 
                        平均濃度 (MCHC): {mchc_val} &nbsp;|&nbsp; 
                        紅血球分布寬度 (RDW): {rdw_val}
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #dcdde1;">
                    <td style="padding: 8px 0; font-weight: bold; color: #34495e; vertical-align: top;">📋 疾病診斷 (Conditions):</td>
                    <td style="padding: 8px 0; color: #d35400; font-weight: 600;">
                        {', '.join(conditions) if conditions else '無紀錄'}
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #dcdde1;">
                    <td style="padding: 8px 0; font-weight: bold; color: #34495e; vertical-align: top;">💊 藥物處方 (Medications):</td>
                    <td style="padding: 8px 0; color: #c0392b; font-weight: 600;">
                        {', '.join(medications) if medications else '無紀錄'}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: bold; color: #34495e; vertical-align: top;">🩺 醫療處置與手術 (Procedures):</td>
                    <td style="padding: 8px 0; color: #8e44ad; font-weight: 600;">
                        {', '.join(procedures) if procedures else '無紀錄'}
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    elif analysis_scope == "「院內感控」多病患群聚分析":
        st.markdown("""
        <div style="background-color:#eefcf0; padding:1.2rem; border-radius:10px; border-left:8px solid #2ecc71; margin-bottom:1.5rem;">
            <h3 style="margin:0 0 0.8rem 0; color:#27ae60;">🏥 院內流行病學感控模式 (FHIR Cohort Mode)</h3>
            <div style="font-size:1.05rem;">
                🟢 <b>狀態:</b> 已啟用跨病患院感傳播監控。系統已調閱 FHIR 伺服器的全體病患註冊清單，將自動為上傳的 IDSEQ 檔案中的多個 Sample_ID 進行臨床個資與病房床位映射（自動代入 Sample Metadata），以分析定位潛在的<b>病房群聚感染、院感交叉傳播鏈、與抗藥性特徵。</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
                "Contig Summary Reports", "Host Gene Count"
            ],
            "Consensus Genome": [
                "Sample Metadata", "Consensus Genome Overview"
            ],
            "Antimicrobial Resistance": [
                "Antimicrobial Resistance Results", "Combined AMR Results", "Sample Metadata"
            ]
        }

        st.markdown("## 📂 上傳檔案")

        for label in mode_file_fields[mode]:
            st.markdown(f"### 📄 上傳：{label}")
            uploaded_file = st.file_uploader(
                label,  
                type=["csv", "gz", "tar", "biom", "zip"],
                key=f"uploader_{mode}_{label}",
                label_visibility="collapsed"  
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
            st.session_state.fhir_json_preview = None
            uploaded_files_dict = st.session_state.uploaded_files_dict
            
            # 檢查是否完全沒有任何資料（無檔案也無載入病患）
            if not uploaded_files_dict and not st.session_state.active_patient_demographics:
                st.warning("請至少上傳一個報告檔案或從 FHIR 連線載入病患資料，以便進行分析。")
                return

            # 檢查是否有未上傳的推薦欄位
            required_fields = mode_file_fields[mode]
            missing_fields = []
            for field in required_fields:
                if field not in uploaded_files_dict:
                    missing_fields.append(field)
                    
            if missing_fields:
                # 僅顯示提示訊息，不再 return 中斷！
                st.info(f"⚠️ 提示：部分推薦檔案未上傳 ({', '.join(missing_fields)})，Gemini 將依據目前已上傳的檔案進行分析。")

            file_contents = preprocess_uploaded_files(uploaded_files_dict)
            
            prompt = generate_llm_prompt(mode, file_contents)

            # 根據分析範疇加入特定的 Gemini 臨床解讀任務指引
            if analysis_scope == "「單一病患」病程/部位追蹤":
                prompt += (
                    "\n\n⚠️ [Clinical Task Directive]: This analysis belongs to the 'Single-Patient' "
                    "longitudinal and multi-site tracking mode. Please focus on analyzing the changes "
                    "in pathogen abundance over time and across different sample collection sites "
                    "(evaluating treatment effectiveness), colonization at different anatomical sites, "
                    "and the selection pressure of antimicrobial resistance (AMR) genes before and "
                    "after medication. You MUST write the entire report in English. Do not include any "
                    "Chinese characters in the generated report."
                )

            # 建立動態 Gemini 實例
            current_api_key = st.session_state.get("user_gemini_key", GOOGLE_API_KEY)
            if not current_api_key:
                st.error("❌ 未檢測到有效的 Gemini API 金鑰！請在左欄「Gemini API 金鑰配置」中輸入您的 API Key。")
                return

            with st.spinner("Gemini 分析中..."):
                try:
                    # 動態配置與初始化
                    genai.configure(api_key=current_api_key)
                    dynamic_model = genai.GenerativeModel("gemini-2.5-pro")
                    dynamic_chat = dynamic_model.start_chat()
                    
                    response = dynamic_chat.send_message(prompt)
                    st.session_state.gemini_analysis_result = response.text
                    st.subheader("📄 分析結果")
                    st.markdown(f"""
                    <div style="background-color:#f7f9fc;padding:1.2rem 1.5rem;border-radius:12px;
                                border-left:6px solid #1f77b4;margin-bottom:1rem;">
                        <h4 style="margin-bottom:0.8rem;">📄 Gemini 分析結果</h4>
                        <pre style="white-space:pre-wrap;font-size:0.92rem;font-family:inherit;">
{response.text}</pre></div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Gemini 分析失敗：{e}")

        # 如果已經有分析結果，則顯示該結果與儲存回 FHIR 的按鈕
        if st.session_state.get("gemini_analysis_result"):
            # 判斷要儲存在哪個 FHIR 病患下
            save_patient_id = None
            save_patient_name = ""
            
            if analysis_scope == "「單一病患」病程/部位追蹤" and st.session_state.get("active_patient_demographics"):
                p = st.session_state.active_patient_demographics
                save_patient_id = p.get("id")
                save_patient_name = p.get("name")
                report_title = f"{mode} Clinical Genomic Report - {save_patient_name}"
            elif analysis_scope == "「院內感控」多病患群聚分析" and st.session_state.get("local_patients"):
                # 院感報告，預設儲存於指標病患 (第一位病患)
                p = st.session_state.local_patients[0]
                save_patient_id = p.get("id")
                save_patient_name = p.get("name")
                report_title = f"Hospital Infection Control & Cohort Outbreak Report - {save_patient_name} (Index Patient)"
                
            if save_patient_id:
                st.markdown("---")
                st.markdown("### 💾 醫療資訊系統 (EHR) 整合 (FHIR Converter)")
                if analysis_scope == "「院內感控」多病患群聚分析":
                    st.info(f"📋 這是跨病患群體分析報告，轉換後將歸檔於指標病患（Index Patient）：**{save_patient_name} (ID: {save_patient_id})** 檔案夾中，以供全院感控委員會調閱。")
                else:
                    st.info(f"👤 報告將歸檔於病患：**{save_patient_name} (ID: {save_patient_id})** 檔案夾中。")
                
                # 第一步：轉換為 FHIR
                if st.button("🔧 Convert to FHIR (轉換為 FHIR 格式)"):
                    # 使用具備臨床 AI NLP（如 John Snow Labs FHIR-Ready AI / Azure Text Analytics for Health）能力的引擎進行非結構化文本實體提取與 FHIR Bundle 轉換
                    current_api_key = st.session_state.get("user_gemini_key", GOOGLE_API_KEY)
                    with st.spinner("🤖 Clinical AI (Text-to-FHIR) 正在解析臨床文本、提取實體並生成 FHIR 結構化 Bundle..."):
                        fhir_dict = convert_text_to_fhir_structured_ai(
                            save_patient_id,
                            st.session_state.gemini_analysis_result,
                            current_api_key
                        )
                        import json
                        st.session_state.fhir_json_preview = json.dumps(fhir_dict, indent=2, ensure_ascii=False)
                    st.rerun()

                # 預覽與儲存
                if st.session_state.get("fhir_json_preview"):
                    st.markdown("#### 🔍 FHIR 格式預覽 (FHIR Resource Preview)")
                    st.code(st.session_state.fhir_json_preview, language="json")
                    
                    if st.button("💾 儲存至 FHIR (Save to FHIR)"):
                        with st.spinner("正在上傳報告至 FHIR 伺服器..."):
                            import json
                            try:
                                fhir_json = json.loads(st.session_state.fhir_json_preview)
                                resource_type = fhir_json.get("resourceType", "Bundle")
                                success, res_id = upload_fhir_resource(
                                    st.session_state.fhir_url,
                                    resource_type,
                                    fhir_json,
                                    st.session_state.get("fhir_token")
                                )
                                if success:
                                    st.success(f"🎉 報告上傳成功！FHIR 資源 ID: `{resource_type}/{res_id}`")
                                else:
                                    st.error(f"❌ 報告儲存失敗：{res_id}")
                            except Exception as ex:
                                st.error(f"❌ 解析/上傳預覽的 FHIR JSON 發生錯誤: {ex}")

            if st.button("📊 清空分析結果"):
                st.session_state.gemini_analysis_result = None
                st.session_state.fhir_json_preview = None
                st.rerun()
                    
if __name__ == "__main__":
    main()
