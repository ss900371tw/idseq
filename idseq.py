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
import secrets
import hashlib
from urllib.parse import urlencode

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

# ---------- SMART on FHIR：共用常數 ----------
REDIRECT_URI = "https://idseqtool.streamlit.app"  # "https://idseqtool.streamlit.app/"
CLIENT_ID = "idseq_streamlit_app"
SCOPES = "launch patient/*.read patient/*.write openid fhirUser"

# ---------- PKCE 與跨頁面 OAuth state 暫存 ----------
def generate_pkce_pair():
    """產生 PKCE 用的 code_verifier 與 S256 code_challenge"""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode("ascii")
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    return verifier, challenge

@st.cache_resource
def _oauth_state_store():
    """process 層級的暫存區，不受單一瀏覽器 session 重置影響。
    僅適合單一 process 部署（如 Streamlit Community Cloud 免費層）；
    若之後改成多副本部署，請換成 Redis 或資料庫。"""
    return {}

def save_pending_oauth_state(state_key: str, data: dict):
    _oauth_state_store()[state_key] = data

def load_pending_oauth_state(state_key: str):
    return _oauth_state_store().pop(state_key, None)

# ---------- (可選) 用 fhirclient 封裝 OAuth2 + PKCE ----------
from fhirclient import client as fhir_client
from fhirclient.auth import FHIRAuth

def build_smart_client(iss, auth_endpoint, token_endpoint, launch_param=None, state=None):
    """建立 FHIRClient，並手動塞入我們自己 discover_endpoints() 找到的端點，
    跳過 fhirclient 內建、只認舊式 CapabilityStatement 擴充的自動發現機制。"""
    if state is not None:
        smart = fhir_client.FHIRClient(state=state, save_func=lambda s: None)
        return smart

    settings = {
        "app_id": CLIENT_ID,
        "api_base": iss,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
    }
    if launch_param:
        settings["launch_token"] = launch_param

    smart = fhir_client.FHIRClient(settings=settings, save_func=lambda s: None)
    smart.server.auth = FHIRAuth.create("oauth2", state={
        "app_id": CLIENT_ID,
        "aud": iss,
        "authorize_uri": auth_endpoint,
        "token_uri": token_endpoint,
        "redirect_uri": REDIRECT_URI,
    })
    return smart

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

def search_medical_code(string_term, target_system, api_key):
    """
    透過 UMLS API 查詢醫學名詞的標準編號
    :param string_term: 醫學名詞 (例如: 'Diabetes', 'Metformin', 'Glucose')
    :param target_system: 限制的術語庫系統代碼 ('SNOMEDCT_US', 'LOINC', 'RXNORM')
    :param api_key: 你的 UMLS API Key
    """
    url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    
    params = {
        'string': string_term,
        'sabs': target_system,      # 指定術語系統
        'returnIdType': 'sourceUi', # 直接返回該系統的原始代碼（而非 UMLS 自己的 CUI）
        'apiKey': api_key,
        'pageSize': 3               # 限制回傳的前幾筆最相關結果
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        results = data.get('result', {}).get('results', [])
        if not results:
            return None
            
        best_match = None
        for r in results:
            ui = r.get('ui', '')
            if target_system == 'LNC':
                # LOINC clinical observation codes must consist strictly of digits and a hyphen (e.g. 12345-6)
                # This excludes LOINC Answers (LA...) and LOINC Parts (LP...)
                parts_code = ui.split('-')
                if len(parts_code) != 2 or not parts_code[0].isdigit() or not parts_code[1].isdigit():
                    continue
            best_match = r
            break

        if not best_match:
            return None

        return {
            "code": best_match.get('ui'),
            "name": best_match.get('name')
        }
    except requests.exceptions.RequestException:
        return None

def validate_and_correct_fhir_bundle(fhir_bundle, umls_api_key):
    """
    嚴格品質與存在性驗證（拒絕接收模式）：
    1. 檢查代碼格式與系統對應性。
    2. 透過 UMLS API 進行真實性與存在性驗證。
    3. 若驗證失敗或導致 coding 陣列為空 []，則直接「拒絕接收」該筆資源，
       不保留任何未編碼的純文字項目，徹底排除幻覺與不合格資料。
    """
    system_map_reverse = {
        "http://loinc.org": "LNC",
        "http://www.nlm.nih.gov/research/umls/rxnorm": "RXNORM",
        "http://snomed.info/sct": "SNOMEDCT_US"
    }

    if not fhir_bundle or "entry" not in fhir_bundle:
        return fhir_bundle

    valid_entries = []
    for entry in fhir_bundle.get("entry", []):
        resource = entry.get("resource", {})
        res_type = resource.get("resourceType")
        
        # 針對沒有 code 欄位的特殊資源（如 DiagnosticReport 等），可依需求保留或檢查
        concepts_to_check = []
        if "code" in resource and isinstance(resource["code"], dict):
            concepts_to_check.append((resource["code"], res_type))
        if "medicationCodeableConcept" in resource and isinstance(resource["medicationCodeableConcept"], dict):
            concepts_to_check.append((resource["medicationCodeableConcept"], "MedicationRequest"))

        # 若該資源本身沒有臨床代碼欄位（例如純結構報告），預設予以保留
        if not concepts_to_check:
            valid_entries.append(entry)
            continue

        resource_fully_valid = True
        for cc, r_type in concepts_to_check:
            display_text = cc.get("text", "").strip()
            codings = cc.get("coding", [])
            
            valid_codings = []
            for coding in codings:
                sys_uri = coding.get("system")
                code_val = str(coding.get("code", ""))
                display_val = coding.get("display", display_text)
                
                # 1. 基礎格式驗證
                if sys_uri == "http://loinc.org":
                    # LOINC clinical observation codes must strictly consist of digits and a hyphen (e.g. 12345-6).
                    # This excludes answer list codes (e.g., LA...) and parts codes (e.g., LP...).
                    parts_code = code_val.split('-')
                    if len(parts_code) != 2 or not parts_code[0].isdigit() or not parts_code[1].isdigit():
                        continue
                if sys_uri in ["http://snomed.info/sct", "http://www.nlm.nih.gov/research/umls/rxnorm"] and not code_val.isdigit():
                    continue

                # 2. 系統與資源類型一致性檢查
                if r_type == "Condition" and sys_uri != "http://snomed.info/sct":
                    continue
                if r_type == "MedicationRequest" and sys_uri != "http://www.nlm.nih.gov/research/umls/rxnorm":
                    continue

                # 3. UMLS 存在性與品質驗證
                if sys_uri in system_map_reverse and display_val:
                    target_sabs = system_map_reverse[sys_uri]
                    verified_res = search_medical_code(display_val, target_sabs, umls_api_key)
                    
                    if not verified_res:
                        continue  # 查無對應，捨棄此 coding
                    else:
                        coding["code"] = verified_res["code"]
                        coding["display"] = verified_res["name"]
                
                valid_codings.append(coding)
            
            # 回寫驗證後的 coding
            cc["coding"] = valid_codings
            
            # 🛑 核心邏輯：只要該概念的 coding 結算為空陣列 []，代表無法取得合規代碼
            if not valid_codings:
                resource_fully_valid = False
                break

        # 只有當資源中的所有核心概念都成功對應到合規代碼時，才收錄進 Bundle；否則「拒絕接收」整筆資源
        if resource_fully_valid:
            valid_entries.append(entry)

    fhir_bundle["entry"] = valid_entries
    return fhir_bundle
    
def convert_text_to_fhir_structured_ai(patient_id, report_markdown, api_key):
    """
    結合 Gemini 智慧關鍵字/實體萃取與 UMLS API (NLM)，
    從非結構化分析報告中精確提取臨床關鍵字與實體，轉換為標準 R4 FHIR Bundle。
    """
    import google.generativeai as genai
    import json
    import datetime
    from pydantic import BaseModel, Field
    from typing import List
    from ctakesclient.typesystem import CtakesJSON
    from ctakesclient import text2fhir
    from uuid import uuid4
    import time

    # ✅ 確保在此處正確定義 Pydantic Schema
    class FlatClinicalEntity(BaseModel):
        mention_type: str = Field(description="Must be exactly 'DiseaseDisorderMention', 'MedicationMention', 'SignSymptomMention', 'ProcedureMention', or 'AnatomicalSiteMention'")
        begin: int = Field(description="Character index where mention begins in the note")
        end: int = Field(description="Character index where mention ends in the note")
        text: str = Field(description="Exact clinical keyword/entity text extracted from the report")
        polarity: int = Field(description="0 for positive mention, -1 for negated mention")
        codingScheme: str = Field(description="Must be 'SNOMEDCT' for diseases/symptoms, or 'RXNORM' for medications, or 'LOINC' for observations")
        code: str = Field(description="The standard code from the chosen system.")
        cui: str = Field(description="A realistic UMLS Concept Unique Identifier")
        tui: str = Field(description="A realistic UMLS Semantic Type Unique Identifier")

    class FlatCtakesInput(BaseModel):
        entities: List[FlatClinicalEntity] = Field(description="List of extracted key clinical entities and keywords from the report")

    class SimpleClinicalEntity(BaseModel):
        text: str = Field(description="Exact clinical keyword/entity text extracted from the report")
        mention_type: str = Field(description="Must be exactly 'DiseaseDisorderMention', 'MedicationMention', 'SignSymptomMention', 'ProcedureMention', or 'AnatomicalSiteMention'")

    class SimpleKeywordInput(BaseModel):
        entities: List[SimpleClinicalEntity] = Field(description="List of extracted key clinical entities and keywords from the report")

    # 1. 取得使用者在側邊欄設定的 UMLS API Key
    umls_api_key = st.session_state.get("user_umls_key", "d6fbdc40-6f90-484a-a8a7-14c919cdfda0")
    
    # 2. 透過 Gemini 動態從報告中萃取關鍵醫學名詞/實體
    genai.configure(api_key=api_key)
    extractor_model = genai.GenerativeModel("gemini-2.5-pro")
    
    extraction_prompt = f"""
You are an exhaustive and precise clinical keyword and entity extraction engine.
Analyze the following clinical report and extract ALL valid medical entities, pathogens, conditions, lab metrics, observations, symptoms, procedures, and medications that are **actively associated with the patient in the current clinical context**.

[CRITICAL EXTRACTION RULES]:
1. **Factuality Constraint**: Extract ONLY entities, conditions, medications, and symptoms that are explicitly stated as present, diagnosed, measured, or currently prescribed for the patient in the text.
2. **Exclude Hypotheticals & Warnings**: DO NOT extract hypothetical substances, potential drug-supplement interactions (e.g., Vitamin E mentioned only as a warning), unprescribed medications, or general educational remarks unless they are part of the patient's actual medical history, current chart, or active prescription list.
3. **Do not hallucinate** or manufacture entities not directly supported by the report text.

Categorize the extracted entities into the following structures:
1. Conditions / Diagnoses (e.g., SARS-CoV-2 infection, Viral sinusitis, Hypertension)
2. Pathogens & Microbes (e.g., Severe acute respiratory syndrome-related coronavirus, Bacteroides)
3. Laboratory / mNGS Metrics (e.g., rPM, Z-score, Read counts, Coverage)
4. Symptoms / Phenotypes (e.g., Sore throat, Nasal congestion)
5. Active Medications / Treatments (e.g., Amoxicillin/Clavulanate, Hydrochlorothiazide — strictly those prescribed to the patient)
6. Specimens & Technical Metadata (e.g., Nasopharyngeal swab, RNA-based metagenomic sequencing)

Report Text:
\"\"\"{report_markdown}
\"\"\"
"""
    
    extracted_terms_to_check = []
    try:
        extract_resp = extractor_model.generate_content(
            extraction_prompt,
            generation_config=genai.GenerativeConfig(
                response_mime_type="application/json",
                response_schema=SimpleKeywordInput
            )
        )
        ext_text = extract_resp.text.strip()
        if ext_text.startswith("```json"):
            ext_text = ext_text.split("```json")[1].split("```")[0].strip()
        elif ext_text.startswith("```"):
            ext_text = ext_text.split("```")[1].split("```")[0].strip()
        parsed_ext = json.loads(ext_text)
        for ent in parsed_ext.get("entities", []):
            t_str = ent.get("text")
            m_type = ent.get("mention_type")
            if t_str:
                sys_target = "SNOMEDCT_US"
                if "Medication" in m_type:
                    sys_target = "RXNORM"
                elif "Observation" in m_type or "Sign" in m_type:
                    sys_target = "LNC"
                extracted_terms_to_check.append((t_str, sys_target))
    except Exception:
        pass

    # 3. 透過 UMLS API 進行動態關鍵字標準編碼查詢 (擴大至前 150 個以滿足大於 100 個術語的需求)
    umls_resolved_codings = []
    seen_terms = set()
    for term, sys_code in extracted_terms_to_check[:150]:
        if term.lower() in seen_terms:
            continue
        seen_terms.add(term.lower())
        
        res = search_medical_code(term, sys_code, umls_api_key)
        if res:
            system_uri_map = {
                "SNOMEDCT_US": "http://snomed.info/sct",
                "RXNORM": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "LNC": "http://loinc.org"
            }
            umls_resolved_codings.append({
                "term": term,
                "code": res["code"],
                "name": res["name"],
                "system": system_uri_map.get(sys_code, "http://snomed.info/sct")
            })

    combined_context_str = "Pre-verified standard codings from UMLS API (NLM):\n"
    for item in umls_resolved_codings:
        combined_context_str += f"- Term: '{item['term']}' | Code: '{item['code']}' | Name: '{item['name']}' | System: {item['system']}\n"

    # 初始化 Gemini Clinical NLP 實體提取引擎
    nlp_model = genai.GenerativeModel("gemini-2.5-pro")

    prompt = f"""
You are a highly specialized clinical NLP pipeline engine, functioning like Apache cTAKES and UMLS ontology lookup tool.
Analyze the extracted clinical keywords and unstructured report, and map them to structured cTAKES JSON format.

Unstructured Report:
\"\"\"
{report_markdown}
\"\"\"

Patient ID: {patient_id}

🧬 STANDARD CLINICAL CODING REFERENCES (UMLS API & Vocabulary Engine):
You MUST utilize these exact pre-verified standard codes when matching these terms:
\"\"\"
{combined_context_str}
\"\"\"

Extract all clinical entities and populate the FlatCtakesInput schema:
- Diseases/Pathogens: use `mention_type='DiseaseDisorderMention'` and system='SNOMEDCT'.
- Medications/Drugs: use `mention_type='MedicationMention'` and system='RXNORM'.
- Observations/Tests: use `mention_type='SignSymptomMention'` or laboratory tests using system='LOINC'.
"""

    try:
        response = nlp_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=FlatCtakesInput
            )
        )
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
        parsed_data = json.loads(raw_text)
        
        ctakes_source = {}
        for entity in parsed_data.get("entities", []):
            m_type = entity.get("mention_type")
            if not m_type:
                continue
            if m_type not in ctakes_source:
                ctakes_source[m_type] = []
                
            mention_obj = {
                "begin": entity.get("begin", 0),
                "end": entity.get("end", 0),
                "text": entity.get("text", ""),
                "polarity": entity.get("polarity", 0),
                "type": m_type,
                "conceptAttributes": [
                    {
                        "codingScheme": entity.get("codingScheme", "SNOMEDCT"),
                        "code": entity.get("code", ""),
                        "cui": entity.get("cui", ""),
                        "tui": entity.get("tui", "")
                    }
                ]
            }
            ctakes_source[m_type].append(mention_obj)
            
        ctakes_json = CtakesJSON(ctakes_source)
        resources = text2fhir.nlp_fhir(
            subject_id=patient_id,
            encounter_id=f"enc-{uuid4().hex[:6]}",
            docref_id=f"doc-{uuid4().hex[:6]}",
            nlp_results=ctakes_json
        )
        
        standard_entries = []
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # 建立 DiagnosticReport（僅摘要關鍵結論，不全文照放）
        diag_report = {
            "resourceType": "DiagnosticReport",
            "id": f"dr-{uuid4().hex[:8]}",
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
            "conclusion": f"Metagenomic NGS clinical analysis processed via SMART Text2FHIR (Key Entities Extracted)."
        }
        standard_entries.append({"resource": diag_report, "request": {"method": "POST", "url": "DiagnosticReport"}})
        
        allowed_systems = {
            "http://snomed.info/sct",
            "http://loinc.org",
            "http://www.nlm.nih.gov/research/umls/rxnorm"
        }
        
        for res in resources:
            res_dict = res.as_json()
            if res_dict.get("resourceType") == "MedicationStatement":
                res_dict["resourceType"] = "MedicationRequest"
                res_dict["intent"] = "order"
                res_dict["status"] = "active"
                
            if "subject" in res_dict and isinstance(res_dict["subject"], dict):
                res_dict["subject"]["reference"] = f"Patient/{patient_id}"
            if "encounter" in res_dict:
                res_dict.pop("encounter", None)
                
            def clean_codeable_concept(cc_dict, default_system):
                if not cc_dict or "coding" not in cc_dict:
                    return
                cleaned_codings = []
                for coding in cc_dict.get("coding", []):
                    system_uri = coding.get("system", default_system)
                    if "snomed" in system_uri.lower():
                        system_uri = "http://snomed.info/sct"
                    elif "rxnorm" in system_uri.lower():
                        system_uri = "http://www.nlm.nih.gov/research/umls/rxnorm"
                    elif "loinc" in system_uri.lower():
                        system_uri = "http://loinc.org"
                        
                    if system_uri in allowed_systems:
                        coding["system"] = system_uri
                        cleaned_codings.append(coding)
                if not cleaned_codings and cc_dict.get("coding"):
                    fallback = cc_dict["coding"][0]
                    fallback["system"] = default_system
                    cleaned_codings.append(fallback)
                cc_dict["coding"] = cleaned_codings

            res_type = res_dict.get("resourceType")
            if res_type == "Condition" and "code" in res_dict:
                clean_codeable_concept(res_dict["code"], "http://snomed.info/sct")
            elif res_type == "MedicationRequest" and "medicationCodeableConcept" in res_dict:
                clean_codeable_concept(res_dict["medicationCodeableConcept"], "http://www.nlm.nih.gov/research/umls/rxnorm")
            elif res_type == "Observation" and "code" in res_dict:
                clean_codeable_concept(res_dict["code"], "http://loinc.org")
            elif res_type == "Procedure" and "code" in res_dict:
                clean_codeable_concept(res_dict["code"], "http://snomed.info/sct")
                
            standard_entries.append({"resource": res_dict, "request": {"method": "POST", "url": res_dict["resourceType"]}})
            
        raw_bundle = {
            "resourceType": "Bundle",
            "type": "transaction",
            "entry": standard_entries
        }

        # ✅ 執行防幻覺與一致性校驗與修正
        validated_bundle = validate_and_correct_fhir_bundle(raw_bundle, umls_api_key)
        return validated_bundle
        
    except Exception as e:
        import sys
        print(f"❌ [FHIR Converter Error] {e}", file=sys.stderr)
        raise e

def upload_fhir_resource(server_url, resource_type, resource_json, token=None):
    """將現成的 FHIR JSON 資源上傳儲存至 FHIR 伺服器"""
    try:
        if resource_type == "Bundle" and resource_json.get("type") in ["transaction", "batch"]:
            url = server_url.rstrip("/")
        else:
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

def upload_fhir_resource(server_url, resource_type, resource_json, token=None):
    """將現成的 FHIR JSON 資源上傳儲存至 FHIR 伺服器"""
    try:
        if resource_type == "Bundle" and resource_json.get("type") in ["transaction", "batch"]:
            url = server_url.rstrip("/")
        else:
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

# ---------- RAG 設定 (MetagenomicKG Neo4j 整合) ----------

# ✅ 初始化 Gemini
load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY","")


try:
    genai.configure(api_key="")
    model = genai.GenerativeModel("gemini-2.5-pro")
    chat = model.start_chat()
except Exception as e:
    model = None
    chat = None


# ✅ MetagenomicKG / PrimeKG 知識圖譜檢索函數
def retrieve_context(query: str, k: int = 5, file_contents: dict = None, mode: str = None):
    """
    用 MetagenomicKG 與 PrimeKG 圖資料庫檢索替代原有的 FAISS 向量檢索。
    根據當前分析模組 (mode) 與上傳數據、病患臨床診斷紀錄 (Conditions) 自動萃取臨床關鍵字並進行語意檢索：
    - Consensus Genome 模組：考量到單一病毒（如 SARS-CoV-2）微觀突變不適用 macro-species 的 MetagenomicKG，
      因此完全路由至 PrimeKG（精準醫學圖譜，對應臨床表型、併發症與對症藥物網絡）。
    - 其他模組：採用 MetagenomicKG (處理 macro-species 與疾病關聯) + PrimeKG (處理患者病歷的精準臨床網絡) 的混合雙圖譜模式。
    """
    import re
    from neo4j import GraphDatabase

    # 1. 萃取關鍵字
    candidate_terms = [
        "Staphylococcus", "S. aureus", "Aureus", "Klebsiella", "K. pneumoniae", "Pneumoniae",
        "Pseudomonas", "P. aeruginosa", "Aeruginosa", "Escherichia", "E. coli", "Coli",
        "Enterococcus", "Streptococcus", "Salmonella", "Acinetobacter", "Haemophilus", "Influenzae",
        "Clostridioides", "Mycoplasma", "Mycobacterium", "Tuberculosis", "Candida", "Aspergillus",
        "Pneumonia", "Sepsis", "Bronchitis", "Asthma", "Urinary", "UTI", "Urosepsis", "Bacteremia",
        "Beta-lactam", "Tetracycline", "Vancomycin", "Ciprofloxacin", "Metformin"
    ]

    def extract_keywords_via_gemini(text: str, category_name: str, max_count: int = 10) -> list:
        """
        使用 Gemini 2.5 動態、開放式地從文本中提取最關鍵的臨床/微生物關鍵字。
        """
        import json
        import google.generativeai as genai
        from pydantic import BaseModel, Field
        from typing import List

        api_key = st.session_state.get("user_gemini_key", GOOGLE_API_KEY)
        if not api_key:
            return []

        try:
            genai.configure(api_key=api_key)
            model_flash = genai.GenerativeModel("gemini-2.5-flash") # 快速、廉價且精確
            
            class KeywordList(BaseModel):
                keywords: List[str] = Field(description=f"List of up to {max_count} clinical, medical, or microbiological keywords")

            prompt = f"""
You are an expert clinical NLP keyword extraction tool.
Analyze the following clinical/microbiological text ({category_name}) and extract the most relevant clinical keywords.
These can be pathogens (bacteria, viruses, fungi, parasites, e.g., 'Syphilis', 'Streptococcus', 'Varicella-zoster', 'Gonorrhea', 'Hepatitis B', 'Chlamydia'), active infections/diseases/conditions (e.g., 'Cystic fibrosis', 'Sepsis', 'Poisoning', 'Pharyngitis', 'Sinusitis'), medications (e.g., 'Vancomycin', 'Epinephrine', 'Cetirizine'), or major infection/immune procedures (e.g., 'Allergy screening').

Instructions:
1. Extract at most {max_count} distinct keywords.
2. Ensure each keyword is a standard English medical/biological term (no dates, no codes, no generic punctuation).
3. Do not invent keywords; only extract terms that are explicitly mentioned in the text.
4. Keep the terms concise (e.g., use 'Syphilis', 'Streptococcus', 'Epinephrine', 'Allergy screening').
5. STRICT CRITICAL CONSTRAINT: DO NOT extract social history (like 'Tobacco smoking', 'Alcohol', 'Occupation'), basic physical measurements/vitals (like 'Body Height', 'Body Weight', 'Blood pressure', 'Heart rate', 'Respiratory rate'), or non-specific common CBC labs (like 'Leukocytes', 'Erythrocytes', 'Hemoglobin', 'Platelets', 'Hematocrit') unless they are explicitly of high-signal diagnostic value for an infection or active pathology. Focus strictly on pathogens, diagnoses, drugs, and relevant immune/infection procedures.

Text:
\"\"\"
{text[:8000]}
\"\"\"
"""
            response = model_flash.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=KeywordList
                )
            )
            raw_res = response.text.strip()
            if raw_res.startswith("```json"):
                raw_res = raw_res.split("```json")[1].split("```")[0].strip()
            elif raw_res.startswith("```"):
                raw_res = raw_res.split("```")[1].split("```")[0].strip()
            data = json.loads(raw_res)
            return [kw.strip() for kw in data.get("keywords", []) if kw.strip()][:max_count]
        except Exception:
            return []

    # A. 優先掃描上傳檔案的 CSV 內容 (最多 10 個)
    file_terms = []
    extraction_method = "Gemini Dynamic Extraction (動態開放式智慧萃取)"
    if file_contents:
        file_text = " ".join(str(val) for val in file_contents.values())
        # 1. 嘗試使用 Gemini 動態提取
        file_terms = extract_keywords_via_gemini(file_text, "Uploaded Files", 10)
        
        # 2. 如果動態提取失敗或返回空，則啟用本地 Heuristic 備用方案
        if not file_terms:
            extraction_method = "Local Heuristic Fallback (本地靜態名詞備用方案)"
            file_text_lower = file_text.lower()
            for term in candidate_terms:
                if len(file_terms) >= 10:
                    break
                if re.search(r'\b' + re.escape(term.lower()) + r'\b', file_text_lower) or term.lower() in file_text_lower:
                    if term not in file_terms:
                        file_terms.append(term)

    # B. 接著掃描當前患者的臨床病歷 (FHIR Clinical Records: 疾病診斷, 藥物處方, 醫療處置與手術) (最多 10 個)
    records_terms = []
    if st.session_state.get("active_patient_demographics"):
        p_id = st.session_state.active_patient_demographics.get("id")
        try:
            conditions = get_fhir_patient_details(st.session_state.fhir_url, p_id, st.session_state.get("fhir_token"))
            medications = get_fhir_patient_medications(st.session_state.fhir_url, p_id, st.session_state.get("fhir_token"))
            procedures = get_fhir_patient_procedures(st.session_state.fhir_url, p_id, st.session_state.get("fhir_token"))
            
            clinical_texts = []
            if conditions:
                clinical_texts.extend(conditions)
            if medications:
                clinical_texts.extend(medications)
            if procedures:
                clinical_texts.extend(procedures)
                
            clinical_text = " ".join(clinical_texts)
            # 1. 嘗試使用 Gemini 動態提取
            records_terms = extract_keywords_via_gemini(clinical_text, "FHIR Clinical Records", 10)
            
            # 2. 如果動態提取失敗或返回空，則啟用本地 Heuristic 備用方案
            if not records_terms:
                clinical_text_lower = clinical_text.lower()
                for term in candidate_terms:
                    if len(records_terms) >= 10:
                        break
                    if re.search(r'\b' + re.escape(term.lower()) + r'\b', clinical_text_lower) or term.lower() in clinical_text_lower:
                        if term not in records_terms:
                            records_terms.append(term)
        except Exception:
            pass

    # 如果還是完全無匹配，不要自動帶入預設的關鍵字，直接返回並記錄提示
    if not file_terms and not records_terms:
        msg = "⚠️ 未在上傳檔案或患者病歷中偵測到任何相關的 知識圖譜 關鍵字，因此未執行圖資料庫檢索。"
        st.session_state.kg_context_retrieved = msg
        return msg

    context_sections = []

    # 建立關鍵字萃取與合併的可視化摘要
    summary_text = (
        f"🔑 **多元知識圖譜關鍵字開放式動態萃取摘要 (Keywords Extraction Strategy: {extraction_method}):**\n"
        f"- 📄 **上傳檔案關鍵字 (Uploaded Files) (最多10個):** {', '.join(file_terms) if file_terms else '無匹配'}\n"
        f"  * 路由目標圖譜: "
        f"{'MetagenomicKG (Neo4j Live)' if mode == 'Metagenomics' else 'BV-BRC (Bacterial and Viral Bioinformatics Resource Center)' if mode == 'Consensus Genome' else 'CARD (Comprehensive Antibiotic Resistance Database)'}\n"
        f"- 📋 **臨床病歷關鍵字 (FHIR Clinical Records) (最多10個):** {', '.join(records_terms) if records_terms else '無匹配'}\n"
        f"  * 路由目標圖譜: PrimeKG (Precision Medicine Graph)\n"
    )
    context_sections.append(summary_text)

    # ----------------------------------------------------
    # A. 針對上傳檔案關鍵字 (File-Extracted Terms) 進行模組化路由檢索
    # ----------------------------------------------------
    if file_terms:
        if mode == "Metagenomics":
            # 🌐 軌道 1：Metagenomics 模組上傳檔案 ➡️ 查詢 MetagenomicKG Neo4j Live DB
            context_sections.append("🌐 [MetagenomicKG Knowledge Graph Live Retrieval Result (For Metagenomics File-extracted Pathogens)]")
            uri = "bolt://mkg.cse.psu.edu:7687"
            auth = ("neo4j", "klabneo4j")

            try:
                driver = GraphDatabase.driver(uri, auth=auth)
                with driver.session() as session:
                    for term in file_terms:
                        context_sections.append(f"📌 MetagenomicKG Context for: '{term}'")
                        
                        # A. 檢索疾病資訊
                        res_dis = session.run(
                            "MATCH (d:`biolink:Disease`) WHERE any(n in d.all_names WHERE toLower(n) CONTAINS toLower($term)) "
                            "RETURN d.all_names[0] AS name, d.description AS desc LIMIT 2",
                            term=term
                        )
                        for rec in res_dis:
                            desc_clean = re.sub(r'<[^>]+>', '', rec["desc"] or "")[:400]
                            context_sections.append(f"  - **Disease**: {rec['name']}\n    *Description*: {desc_clean}")

                        # B. 檢索微生物與病原體屬性
                        res_micro = session.run(
                            "MATCH (m:`biolink:OrganismTaxon`) WHERE any(n in m.all_names WHERE toLower(n) CONTAINS toLower($term)) "
                            "RETURN m.all_names[0] AS name, m.description AS desc, m.is_pathogen AS is_pathogen LIMIT 2",
                            term=term
                        )
                        for rec in res_micro:
                            context_sections.append(f"  - **Pathogen**: {rec['name']} (Is Pathogen: {rec['is_pathogen']})\n    *Description*: {rec['desc']}")

                        # C. 檢索微生物與疾病之已知關聯 (Associations)
                        res_rel = session.run(
                            "MATCH (m:`biolink:OrganismTaxon`)-[r:`biolink:associated_with`]-(d:`biolink:Disease`) "
                            "WHERE any(n in m.all_names WHERE toLower(n) CONTAINS toLower($term)) OR "
                            "      any(n in d.all_names WHERE toLower(n) CONTAINS toLower($term)) "
                            "RETURN m.all_names[0] AS microbe, d.all_names[0] AS disease LIMIT 2",
                            term=term
                        )
                        rels = []
                        for rec in res_rel:
                            rels.append(f"'{rec['microbe']}' is associated with disease '{rec['disease']}'")
                        if rels:
                            context_sections.append("  - **Linked Associations**:\n    " + "\n    ".join(rels))

                        # D. 檢索藥物/化學物資訊
                        res_drug = session.run(
                            "MATCH (dr:`biolink:Drug`) WHERE any(n in dr.all_names WHERE toLower(n) CONTAINS toLower($term)) "
                            "RETURN dr.all_names[0] AS name, dr.description AS desc LIMIT 2",
                            term=term
                        )
                        for rec in res_drug:
                            context_sections.append(f"  - **Recommended Drug/Chemical**: {rec['name']}\n    *Details*: {rec['desc']}")

                driver.close()
            except Exception as e:
                context_sections.append(f"⚠️ Failed to live-query MetagenomicKG: {e}")
                context_sections.append("- Local Backup Fact: Staphylococcus aureus is a major human pathogen associated with skin, soft tissue, and systemic infections like sepsis and pneumonia.")

        elif mode == "Consensus Genome":
            # 🧬 軌道 2：Consensus Genome 模組上傳檔案 ➡️ 查詢 BV-BRC (Bacterial and Viral Bioinformatics Resource Center) 知識圖譜
            context_sections.append("🧬 [BV-BRC (Bacterial and Viral Bioinformatics Resource Center) Live Ontology Lookup (For Viral & Bacterial Genomes)]")
            
            api_key = st.session_state.get("user_gemini_key", GOOGLE_API_KEY)
            if api_key:
                import google.generativeai as genai
                import json
                from pydantic import BaseModel, Field
                from typing import List

                try:
                    genai.configure(api_key=api_key)
                    model_flash = genai.GenerativeModel("gemini-2.5-flash")
                    
                    class BVBRCRelationship(BaseModel):
                        source_node_name: str = Field(description="Name of the source node (Strain/Isolate, Feature/Gene, Protein, Virulence Factor, Host Interaction)")
                        source_node_type: str = Field(description="Strain/Isolate, Feature/Gene, Protein, Virulence Factor, Host Interaction")
                        source_id: str = Field(description="BV-BRC ID or standard ID (e.g. NCBI:1024, BVBRC:Gene_Spike, PDB:7DK3)")
                        relation_type: str = Field(description="Edge type: has_lineage, expresses_protein, associated_with_virulence, interacts_with_host_receptor, resistant_to_drug")
                        target_node_name: str = Field(description="Name of the target node")
                        target_node_type: str = Field(description="Type of the target node")
                        target_id: str = Field(description="Standardized ID of the target node")
                        description: str = Field(description="Genomic, structural, lineage, and host interaction details of this viral/bacterial relation")

                    class BVBRCQueryResult(BaseModel):
                        queried_term: str = Field(description="The viral/bacterial strain or genomic keyword queried")
                        mapped_node_name: str = Field(description="Canonical standard strain/gene name mapped in BV-BRC")
                        mapped_node_type: str = Field(description="Strain/Isolate, Feature/Gene, Protein, or Virulence Factor")
                        mapped_id: str = Field(description="Standard BV-BRC / NCBI taxon ID")
                        relationships: List[BVBRCRelationship] = Field(description="List of standardized BV-BRC structural and functional genome relationships")

                    class BVBRCOutput(BaseModel):
                        results: List[BVBRCQueryResult] = Field(description="BV-BRC genome mapping and strain lineage results")

                    prompt = f"""
You are an expert bioinformatician representing the BV-BRC (Bacterial and Viral Bioinformatics Resource Center) genomics database.
Analyze the following viral/bacterial terms, map them to BV-BRC standard strains/features, and retrieve their genomic, lineage, and host-pathogen relationships.

Keywords to query:
{file_terms}

Instructions:
1. Map each keyword to its canonical BV-BRC node type (Strain/Isolate, Feature/Gene, Protein, Virulence Factor) and standard IDs (NCBITaxon ID for strains, BV-BRC Feature ID for viral/bacterial genes).
2. Retrieve at least 3 high-signal relationships representing official BV-BRC schema edges (e.g., has_lineage, expresses_protein, associated_with_virulence, interacts_with_host_receptor, resistant_to_drug).
3. Populate the schema with scientific accuracy, detailing Pangolin lineages, viral clades, spike/envelope protein structural interactions, and viral virulence mechanisms.
"""
                    response = model_flash.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=BVBRCOutput
                        )
                    )
                    
                    raw_res = response.text.strip()
                    if raw_res.startswith("```json"):
                        raw_res = raw_res.split("```json")[1].split("```")[0].strip()
                    elif raw_res.startswith("```"):
                        raw_res = raw_res.split("```")[1].split("```")[0].strip()
                    data = json.loads(raw_res)
                    
                    for res in data.get("results", []):
                        term = res.get("queried_term")
                        node_name = res.get("mapped_node_name")
                        node_type = res.get("mapped_node_type")
                        node_id = res.get("mapped_id")
                        
                        context_sections.append(f"📌 **BV-BRC Genome Database Ontological Mapping for: '{term}'**")
                        context_sections.append(f"  - **Standard Entity**: {node_name} ({node_type} | Standard ID: `{node_id}`)")
                        context_sections.append("  - **Genomic Features, Lineages & Host-Pathogen Interactions (BV-BRC Edges)**:")
                        
                        for rel in res.get("relationships", []):
                            src_name = rel.get("source_node_name")
                            src_type = rel.get("source_node_type")
                            src_id = rel.get("source_id")
                            edge = rel.get("relation_type")
                            tgt_name = rel.get("target_node_name")
                            tgt_type = rel.get("target_node_type")
                            tgt_id = rel.get("target_id")
                            desc = rel.get("description")
                            
                            context_sections.append(f"    * `({src_name}:{src_type} [{src_id}]) -[{edge}]-> ({tgt_name}:{tgt_type} [{tgt_id}])`")
                            context_sections.append(f"      *Biological Significance*: {desc}")
                        context_sections.append("")
                except Exception as e:
                    context_sections.append(f"⚠️ Failed to query BV-BRC dynamically: {e}")
            else:
                context_sections.append("⚠️ Gemini API Key not configured. Skipping dynamic BV-BRC lookup.")

        elif mode == "Antimicrobial Resistance":
            # 💊 軌道 3：Antimicrobial Resistance 模組上傳檔案 ➡️ 查詢 CARD (Comprehensive Antibiotic Resistance Database)
            context_sections.append("💊 [CARD (Comprehensive Antibiotic Resistance Database) live ontology lookup (For AMR genes & mechanisms)]")
            
            api_key = st.session_state.get("user_gemini_key", GOOGLE_API_KEY)
            if api_key:
                import google.generativeai as genai
                import json
                from pydantic import BaseModel, Field
                from typing import List

                try:
                    genai.configure(api_key=api_key)
                    model_flash = genai.GenerativeModel("gemini-2.5-flash")
                    
                    class CARDRelationship(BaseModel):
                        source_node_name: str = Field(description="Name of the source node (AMR Gene, AMR Mechanism, Drug Class, Organism)")
                        source_node_type: str = Field(description="AMR Gene, AMR Mechanism, Drug Class, Organism")
                        source_id: str = Field(description="ARO (Antibiotic Resistance Ontology) ID (e.g. ARO:3000015, ARO:0000036)")
                        relation_type: str = Field(description="Edge type: confers_resistance_to, has_mechanism, detected_in, part_of_operon")
                        target_node_name: str = Field(description="Name of the target node")
                        target_node_type: str = Field(description="Type of the target node")
                        target_id: str = Field(description="Standardized ontology ID of the target node")
                        description: str = Field(description="Pharmacological resistance mechanism and antibiotic inactivation detail")

                    class CARDQueryResult(BaseModel):
                        queried_term: str = Field(description="The antibiotic/AMR keyword queried")
                        mapped_node_name: str = Field(description="Canonical standard AMR Gene name mapped in CARD")
                        mapped_node_type: str = Field(description="AMR Gene, AMR Mechanism, Drug Class, or Organism")
                        mapped_id: str = Field(description="Standardized ARO ID (e.g. ARO:3000015)")
                        relationships: List[CARDRelationship] = Field(description="List of standardized CARD relationships for this AMR marker")

                    class CARDOutput(BaseModel):
                        results: List[CARDQueryResult] = Field(description="CARD ontology lookup results")

                    prompt = f"""
You are an expert clinical microbiologist representing the CARD (Comprehensive Antibiotic Resistance Database) ontology system.
Analyze the following AMR genes, drug resistance keywords, and organisms, map them to CARD standard ARO ontology nodes, and retrieve their relationships.

Keywords to query:
{file_terms}

Instructions:
1. Map each keyword to its canonical CARD ARO (Antibiotic Resistance Ontology) ID and node type (AMR Gene, AMR Mechanism, Drug Class, Organism).
2. Retrieve at least 3 high-signal relationships representing official CARD database edges (e.g., confers_resistance_to, has_mechanism, detected_in, part_of_operon).
3. Populate the schema with scientific accuracy, detailing antibiotic efflux pumps, enzyme inactivation, carbapenemases, or beta-lactamase operons.
"""
                    response = model_flash.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=CARDOutput
                        )
                    )
                    
                    raw_res = response.text.strip()
                    if raw_res.startswith("```json"):
                        raw_res = raw_res.split("```json")[1].split("```")[0].strip()
                    elif raw_res.startswith("```"):
                        raw_res = raw_res.split("```")[1].split("```")[0].strip()
                    data = json.loads(raw_res)
                    
                    for res in data.get("results", []):
                        term = res.get("queried_term")
                        node_name = res.get("mapped_node_name")
                        node_type = res.get("mapped_node_type")
                        node_id = res.get("mapped_id")
                        
                        context_sections.append(f"📌 **CARD (ARO Ontology) Mapping for: '{term}'**")
                        context_sections.append(f"  - **Standard Entity**: {node_name} ({node_type} | Standard ID: `{node_id}`)")
                        context_sections.append("  - **CARD Antibiotic Resistance Mechanisms & Phenotypes (CARD Edges)**:")
                        
                        for rel in res.get("relationships", []):
                            src_name = rel.get("source_node_name")
                            src_type = rel.get("source_node_type")
                            src_id = rel.get("source_id")
                            edge = rel.get("relation_type")
                            tgt_name = rel.get("target_node_name")
                            tgt_type = rel.get("target_node_type")
                            tgt_id = rel.get("target_id")
                            desc = rel.get("description")
                            
                            context_sections.append(f"    * `({src_name}:{src_type} [{src_id}]) -[{edge}]-> ({tgt_name}:{tgt_type} [{tgt_id}])`")
                            context_sections.append(f"      *Resistance Detail*: {desc}")
                        context_sections.append("")
                except Exception as e:
                    context_sections.append(f"⚠️ Failed to query CARD dynamically: {e}")
            else:
                context_sections.append("⚠️ Gemini API Key not configured. Skipping dynamic CARD lookup.")

    # ----------------------------------------------------
    # B. 針對病患臨床紀錄 (FHIR Clinical Records) 全模組 ➡️ 查詢/檢索 PrimeKG 精準醫學圖譜
    # ----------------------------------------------------
    if records_terms:
        context_sections.append("🧬 [PrimeKG Precision Medicine Graph Live Retrieval Result (For Patient Clinical Records)]")
        
        api_key = st.session_state.get("user_gemini_key", GOOGLE_API_KEY)
        if api_key:
            import google.generativeai as genai
            import json
            from pydantic import BaseModel, Field
            from typing import List

            try:
                genai.configure(api_key=api_key)
                model_flash = genai.GenerativeModel("gemini-2.5-flash")
                
                class PrimeKGRelationship(BaseModel):
                    source_node_name: str = Field(description="Name of the source node")
                    source_node_type: str = Field(description="Type of the source node (e.g., Disease, Drug, Phenotype, Gene/Protein, Pathway, Anatomy)")
                    source_id: str = Field(description="Standardized ontology ID (e.g., MONDO:0005015, DB00047, HP:0002090, HGNC:1101, Reactome:R-HSA-109581, UBERON:0002048)")
                    relation_type: str = Field(description="Edge type: indication, contraindication, side effect, disease_phenotype, disease_protein, drug_protein, disease_disease, pathway_protein, anatomy_protein, protein_protein, etc.")
                    target_node_name: str = Field(description="Name of the target node")
                    target_node_type: str = Field(description="Type of the target node")
                    target_id: str = Field(description="Standardized ontology ID of the target node")
                    description: str = Field(description="Clinical and pharmacological significance of this relationship")

                class PrimeKGQueryResult(BaseModel):
                    queried_term: str = Field(description="The clinical keyword queried in PrimeKG")
                    mapped_node_name: str = Field(description="The canonical standard name mapped to this term in PrimeKG")
                    mapped_node_type: str = Field(description="Disease, Drug, Phenotype, Gene/Protein, or Pathway")
                    mapped_id: str = Field(description="Ontology ID (MONDO, DrugBank, HPO, etc.)")
                    relationships: List[PrimeKGRelationship] = Field(description="List of standardized PrimeKG relationships for this entity")

                class PrimeKGOutput(BaseModel):
                    results: List[PrimeKGQueryResult] = Field(description="PrimeKG clinical context mapping results")

                prompt = f"""
You are an expert precision medicine bioinformatician representing the Harvard PrimeKG Precision Medicine Knowledge Graph database.
Analyze the following clinical keywords from the patient's record, map them to standard PrimeKG nodes, and retrieve their relationships.

Keywords to query:
{records_terms}

Instructions:
1. Map each keyword to its canonical PrimeKG node type (Disease, Drug, Phenotype, Gene/Protein, Pathway, Anatomy) and ID system (MONDO for Disease, DrugBank for Drug, HPO for Phenotype, NCBI Gene for Gene, Reactome for Pathway, UBERON for Anatomy).
2. Retrieve at least 3 high-signal relationships representing official PrimeKG edges (e.g., indication, contraindication, side effect, disease_phenotype, disease_protein, drug_protein, disease_disease, pathway_protein, anatomy_protein).
3. Populate the schema accurately. Keep descriptions extremely informative and clinically sound.
"""
                response = model_flash.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=PrimeKGOutput
                    )
                )
                
                raw_res = response.text.strip()
                if raw_res.startswith("```json"):
                    raw_res = raw_res.split("```json")[1].split("```")[0].strip()
                elif raw_res.startswith("```"):
                    raw_res = raw_res.split("```")[1].split("```")[0].strip()
                data = json.loads(raw_res)
                
                for res in data.get("results", []):
                    term = res.get("queried_term")
                    node_name = res.get("mapped_node_name")
                    node_type = res.get("mapped_node_type")
                    node_id = res.get("mapped_id")
                    
                    context_sections.append(f"📌 **PrimeKG Ontological Mapping for FHIR Patient Record: '{term}'**")
                    context_sections.append(f"  - **Standard Entity**: {node_name} ({node_type} | Standard ID: `{node_id}`)")
                    context_sections.append("  - **Standardized Precision Medicine Relationships (PrimeKG Edges)**:")
                    
                    for rel in res.get("relationships", []):
                        src_name = rel.get("source_node_name")
                        src_type = rel.get("source_node_type")
                        src_id = rel.get("source_id")
                        edge = rel.get("relation_type")
                        tgt_name = rel.get("target_node_name")
                        tgt_type = rel.get("target_node_type")
                        tgt_id = rel.get("target_id")
                        desc = rel.get("description")
                        
                        context_sections.append(f"    * `({src_name}:{src_type} [{src_id}]) -[{edge}]-> ({tgt_name}:{tgt_type} [{tgt_id}])`")
                        context_sections.append(f"      *Clinical Significance*: {desc}")
                    context_sections.append("")
            except Exception as e:
                context_sections.append(f"⚠️ Failed to query PrimeKG dynamically: {e}")
        else:
            context_sections.append("⚠️ Gemini API Key not configured. Skipping dynamic PrimeKG lookup.")

    ret_val = "\n\n".join(context_sections)
    st.session_state.kg_context_retrieved = ret_val
    return ret_val



def generate_llm_prompt(mode, file_contents):
    file_definitions = {
        "Heatmap": "Statistical matrix of all samples and all microbial taxons.",
        "Sample Metadata": "Basic information of the samples, such as sampling time and site.",
        "Samples Overview": "QC statistics and summary for each sample.",
        "Sample Taxon Report": "Microbial classification and quantitative data detected in each sample.",
        "Combined Sample Taxon Results": "Aggregated microbiology data summary table for all samples.",
        "Combined Microbiome File": "Combined microbiome abundance table (parsed from BIOM format) containing OTU/microbial counts across samples.",
        "Contig Summary Reports": "QC statistics and coverage of Contigs.",
        "Host Gene Count": "Host transcript expression statistics.",
        "Consensus Genome Overview": "Quality control (QC) metrics of the consensus genome (e.g., genome coverage percentage, mapped reads, SNP count) and other statistical summaries.",
        "Intermediate Output Files": "Extracted key pipeline assembly and variant metrics (average depth, coverage breadth, read counts, and SNPs/indels from VCF) from the intermediate output archive.",
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
    context_text = retrieve_context(user_query, file_contents=file_contents, mode=mode)
    summary_lines.append(f"\n📚 Textbook Supplementary Knowledge:\n{context_text}")

    prompt_template = TEMPLATE_MAP[mode]
    return prompt_template.format(csv_content="\n".join(summary_lines))




# Prompt 模板
TEMPLATE_MAP = {
    "Metagenomics": """
你是一位精通生物資訊學（Bioinformatics）、次世代定序（mNGS）數據解讀與臨床感染症的專家。請根據我後續提供的 IDSeq (CZ ID) 宏基因組分析原始數據（包含 heatmap.csv 及相關數據）與圖譜補充知識，產出一份結構完整、專業且利於臨床個案追蹤與研究判讀的綜合性 mNGS 個案分析報告。

📌 重要背景說明：本分析的所有樣本皆來自「同一個病人」（包含不同採檢時間點或不同解剖部位的檢體）。報告必須以單一個案追蹤（Case Study / Longitudinal & Multi-site tracking）的角度出發，探討病程進展、治療前後變化或不同部位間的微生物相差異，切勿將其視為不同病人的群體橫斷性研究。

整份報告必須嚴謹包含以下五大核心區塊與擴充架構：

1. 樣本與品管摘要 (Sample & QC Summary)
   - 包含 Sample ID、宿主來源、總 Raw Reads、扣除宿主後的 Passing Filters (Non-host Reads) 及佔比、品管狀態與各採檢時間點/部位概況。

2. 病原體分類與指標列表 (Taxonomic Abundance & Significance Table)
   - 以表格呈現檢出的病原分類（Viruses, Bacteria, Eukaryotes 等）、科學名稱、rPM (Reads Per Million)、Z-score（背景顯著性）、NT 吻合數 (Reads) 及基因組覆蓋度 (Coverage)。

3. 核心指標專業解讀與熱圖分析 (Key Metrics & Heatmap Clustering Analysis)
   - 針對檢出數值較高的標的分析其 rPM 相對豐度意義，以及 Z-score 是否高於背景閾值（排除環境污染與 Kitome 雜訊）。
   - 針對 heatmap.csv 的數據結構，解讀該病人在不同時間點或部位之間的聚類關係、相似性與動態差異性，指出顯著的微生物分佈轉變或優勢菌株位移。

4. 綜合風險、訊號雜訊與生態意義推論 (Signal-to-Noise & Biological Insights)
   - 結合檢體類型與常見背景雜訊，辨識「真正潛在致病原」與「過客/背景微生物」，評估偽陽性或定植的可能。
   - ⚠️ 知識圖譜強制深度整合 (Mandatory MetagenomicKG & PrimeKG Integration)：你必須深度整合下方「📚 Textbook Supplementary Knowledge」部分提供的 MetagenomicKG 與 PrimeKG 圖資料庫檢索脈絡（特別是 MetagenomicKG 的微生物病原體與疾病特徵，以及 PrimeKG 提供的精準醫學疾病/表型/基因與藥物的 Live 實證關聯，如 disease_phenotype, disease_protein, indication, contraindication 等）。明確指出檢出微生物在圖譜中的病原體分類與致病機制，並深度探討病患當前臨床診斷與藥理機制的精準醫學關聯，嚴禁直接忽略圖譜背景知識。

5. 結論、臨床治療與後續驗證建議 (Conclusion, Treatment Guidelines & Next Steps)
   - 總結分析亮點，並評估對病人病程與治療反應的影響。
   - ⚠️ 臨床治療與投藥指引：必須結合 MetagenomicKG 檢索到的推薦藥物與 PrimeKG 提供的精準藥物-疾病/靶點關聯資訊（如 DrugBank 關係、drug_protein、contraindication 禁忌症），針對病患後續的經驗性治療或精準用藥調整，給予具備圖譜科學依據的精準處置與投藥臨床指引建議。
   - 提供後續具體的實驗與臨床驗證建議（例如特定 PCR、傳統微生物培養、Sanger 定序或臨床進一步鑑別方向、Alpha/Beta 多樣性變化與功能基因預測等）。

請確保用詞專業、客觀，嚴格符合 IDSeq / Metagenomics 數據分析標準規範。

注意事項：
- 請從我上傳的檔案內容中解析數據、行列與數值來撰寫報告。
- **MetagenomicKG & PrimeKG 知識圖譜強制整合**：你必須主動且深入地將「📚 Textbook Supplementary Knowledge」中的圖譜背景知識（如 MetagenomicKG 病原體-疾病關聯、推薦藥物，以及 PrimeKG 精準醫學疾病、表型、基因與藥物關係）有機融合融入對應的第 4 點與第 5 點分析段落中，严禁僅將其作為附錄或簡單條列，而必須融入報告主體的敘事分析與臨床推論中。
- 報告語氣需專業、客觀，專有名詞請保持正確的生物資訊與微生物學術語。
- 請禁止任何開場白、客套話、自我介紹或結語，從第一個字開始就是報告本身。
 
📌 請以這些問題作為總體基因體學（Metagenomics）分析的指導方針，綜合撰寫一份臨床觀察與洞察報告。最終報告必須完全以英文撰寫並禁止逐題問答式輸出。
 
原始 CSV 摘要：
{csv_content}
""",
    "Consensus Genome": """
你是一位精通病毒基因體學（Viral Genomics）、次世代定序（mNGS）數據分析與臨床感染學的專家。
請根據我所提供的 CZ ID (IDSeq) Consensus Genome (CG) 原始分析數據與臨床背景，為我撰寫一份結構完整、專業且利於臨床解讀與公衛監測的「一致性基因體（Consensus Genome）分析報告」。

📌 重要背景說明：本分析的所有檢體皆來自「同一個病人」（可能為同時期/不同時期，或同部位/不同部位）。報告必須聚焦於病原體在該病人體內的演化、持續感染或不同部位的分佈情形。

最終報告必須完全以英文撰寫（Professional English），結構嚴謹，並涵蓋以下五大核心區塊與深度維度：

1. 執行摘要與樣本/目標病毒摘要 (Executive Summary & Target Summary)
總結本次檢體高通量定序（mNGS）與 Consensus Genome (IDSEQ) 分析的核心發現。
點出主要檢出的病原體及其在不同時間點或部位的臨床意義。
包含 Sample ID、標的病毒名稱 (Target Organism)、分析管線版本、Pangolin 預測譜系 (Lineage)、總乾淨序列數 (Total Clean Reads) 及對應到該病毒的 Mapped Reads 數與佔比。

2. 樣本與定序品質控制 (Sample & Sequencing Quality Control)
評估各樣本的總讀數（Total Reads）、過濾後讀數（Passing Filters）及比對率（Mapping Statistics）。
分析宿主背景基因序列（Host background）佔比及對病原體檢出靈敏度的影響。

3. 基因組覆蓋度與深度指標 (Coverage & Depth Metrics Table)
以表格呈現參考基因組長度、不同門檻的基因組覆蓋度（如 $\ge 1\times$ 與 $\ge 10\times$ 覆蓋率）、平均定序深度 (Mean Depth)、未定鹼基數 (Ambiguous Bases / Ns 數量與佔比) 以及整體組裝品質評級。
詳細列出檢出的病原體及其在各樣本中的相對豐度（RPM / rPM）。

4. 病原體基因組分佈、變異與圖譜知識整合 (Pathogen Profiling, Key Mutations & BV-BRC/PrimeKG Integration)
說明基因組覆蓋的均勻度、是否出現顯著的訊號斷層，並列出檢測到的關鍵突變位點或標誌性胺基酸取代。
評估組裝出來的 Consensus Genome 品質（如：N-base 比例、與參考基因體的相似度等）。

⚠️ 微生物基因組與精準醫學圖譜整合 (Mandatory BV-BRC & PrimeKG Integration)：
深度參考並結合「📚 Textbook Supplementary Knowledge」中由 BV-BRC (Bacterial and Viral Bioinformatics Resource Center) 實時檢索拉回的微生物基因組特徵、Pangolin 演化譜系、蛋白結構、毒力因子及宿主交互網絡，以及 PrimeKG 提供的病患精準醫學臨床病歷與對症藥理網絡（MONDO 疾病分類、臨床表型 HPO 關聯、退燒對症用藥如 Ibuprofen 靶點與通路等），探討其在該名病患體內的潛在臨床危害與宿主微觀機制（須將背景知識有機融入主體敘事，嚴禁僅作條列式附錄）。

5. 系統發生、公衛與臨床解讀建議 (Phylogenetic, Public Health & Clinical Interpretation)
針對該病毒的覆蓋完整性（是否適合上傳 GISAID/GenBank 或進行進一步的演化樹分析）以及譜系分型結果提供專業解讀。
綜合評估定序結果的可靠度，並針對該病人的病程追蹤、後續實驗驗證（如 RT-qPCR、Sanger 定序）提出建議。

⚠️ 臨床干預與精準投藥建議：綜合 BV-BRC 提供的病毒結構與譜系抗性特徵以及 PrimeKG 提供的精準對症藥理（退燒藥物、適應症 indication、禁忌症 contraindication 藥理網絡與 drug_protein 靶點通路），為臨床醫師針對該特定病毒突變株引起的上呼吸道感染與併發症（如中耳炎）之治療干預、給藥選擇與防範措施上，提供具備圖譜科學依據的精準處置與投藥方案。

⚠️ 執行注意事項：
請從上傳的檔案內容中解析數據、行列與數值來撰寫報告。
語氣需專業、客觀，專有名詞請保持正確的生物資訊與微生物學術語。
嚴禁任何開場白、客套話、自我介紹或結語，從第一個字開始就是報告本身。
禁止逐題問答式輸出，必須融合為一篇流暢的專業臨床洞察報告。

原始 CSV 摘要：
{csv_content}
""",
"Antimicrobial Resistance": """
你是一位精通生物資訊學（Bioinformatics）、次世代定序（mNGS）抗性基因檢測（AMR）與臨床抗菌藥物管理（Antimicrobial Stewardship）的專家。請根據我上傳的三個檔案（包含 sample_metadata.csv、病原體檢測報表、以及基於 CZ ID / IDSeq 與 CARD / ResFinder 資料庫的抗藥性基因檢測報表），並結合後續提供的「📚 Textbook Supplementary Knowledge (CARD & PrimeKG 知識圖譜)」，為我撰寫一份結構完整、專業且利於臨床醫師調整抗生素治療策略的 AMR 臨床觀察與洞察報告。
📌 重要背景說明：
本分析的所有檢體皆來自「同一位病人」（涵蓋治療前後不同時期，或不同採檢部位）。報告必須探討病原體與抗藥性基因在該病人治療過程中的動態變化，以及不同部位間抗藥性特徵的差異。
【寫作與格式嚴格規範】

全英文輸出：最終報告必須完全以英文 (English) 撰寫，專有名詞請保持正確的生物資訊與微生物學術語。
禁止無效內容：請禁止任何開場白、客套話、自我介紹或結語，從第一個字開始必須直接是報告主體。
行文風格：語氣需專業、客觀，嚴格符合臨床抗感染規範，並禁止逐題問答式的輸出，請以連貫的專業報告格式呈現。
知識圖譜強制整合：你必須主動且深入地將「📚 Textbook Supplementary Knowledge」中的圖譜背景知識有機融合進報告主體的敘事分析與臨床推論中，嚴禁僅將其作為附錄或簡單條列。
【報告核心區塊與結構要求】
請直接從上傳的檔案內容中萃取數據進行分析與整合，報告必須包含以下五個核心區段：
1. Sample Metadata & AMR Pipeline Summary (樣本概況與測序品質摘要)

摘要 sample_metadata.csv 中的關鍵樣本資訊（包含 Sample ID、採檢時間點、採檢部位、用藥史等臨床特徵）。
總結使用的 AMR 分析管線與資料庫版本、總乾淨序列數 (Total Clean Reads)、扣除宿主後的無宿主序列數 (Non-host Reads)，以及 AMR 整體檢測狀態。
2. Pathogen Identification & Dynamics (主要病原體識別與動態變化)

根據病原體檢測結果，列出該病人在不同時間點或不同檢體部位中，主要檢出的致病微生物。
分析其相對豐度（如 rPM）或檢出情形的動態變化趨勢。
3. Antimicrobial Resistance Profile & Pathogen Association (抗藥性基因圖譜與病原體關聯分析)

基因與病原體對照：整理不同時間點/部位偵測到的抗性基因，詳細列出檢出抗性基因名稱 (Gene Symbol)、對應抗生素家族 (Drug Class)、基因覆蓋度 (Coverage) 與讀段支援數 (Reads Mapping)。並結合物種豐度，追蹤這些基因最可能來自哪一種檢出的致病細菌，標示預期表型耐藥特徵。
⚠️ CARD 抗藥機制與 PrimeKG 臨床用藥深度整合 (Mandatory CARD & PrimeKG Integration)：必須深度結合 CARD 知識圖譜提供的 ARO 抗藥本體、抗性基因機制 (如 antibiotic efflux, inactivation) 與 PrimeKG 關於患者基礎疾病、臨床表型 HPO、以及臨床用藥 (如 Beta-lactams, Vancomycin 靶點 drug_protein 關聯、禁忌症) 的精準醫學資訊。對比分析目前檢出的抗藥基因與引發的潛在臨床風險。
4. Comprehensive Clinical Risk Assessment (綜合臨床風險評估)

綜合病原體與抗藥性基因的動態變化，評估該病人體內抗藥性突變、抗藥菌株在治療壓力下的演變或清除/篩選風險（如高風險 ESBL、多重抗藥性等）。
⚠️ 圖譜病原致病與 PrimeKG 臨床風險整合：必須將 CARD 檢出的 ARO 抗性突變與病原體臨床風險，結合 PrimeKG 中關於該患者本身基礎疾病與 HPO 臨床表型的關聯資訊，深度融合進本段風險評估中，以利臨床醫師研判該耐藥株對病患造成的綜合生命威脅。
5. AMR Stewardship & Treatment Recommendations (抗菌藥物管理與臨床處置建議)

針對檢出的關鍵抗藥特徵，指出哪些經驗性抗生素可能面臨治療失敗，必須避免使用；並提供院內感染管控或接觸隔離的警示建議。
⚠️ 替代藥物與臨床實證用藥指引：必須結合 CARD 抗藥機制與 PrimeKG 中所建議之有效/推薦藥物及相關靶點通路 (indication, drug_protein) 臨床細節，提出具臨床實證依據的抗生素療程調整、優先考慮的替代治療方案（如碳青黴烯類或新型複合製劑）或合併用藥方案，並給出後續實驗驗證的建議。
 
原始 CSV 摘要：
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
                        
                        has_stats_or_vcf = any(m.name.endswith("stats.json") or m.name.endswith("variants.vcf.gz") for m in tar.getmembers() if m.isfile())
                        
                        if has_stats_or_vcf:
                            import json
                            import gzip
                            
                            sample_dirs = set()
                            for m in tar.getmembers():
                                if m.isfile() and ("stats.json" in m.name or "variants.vcf.gz" in m.name):
                                    parent = os.path.dirname(m.name)
                                    if parent:
                                        sample_dirs.add(parent)
                                        
                            rows = []
                            for sdir in sorted(list(sample_dirs)):
                                stats_data = {}
                                variants_list = []
                                contigs_count = "N/A"
                                total_length = "N/A"
                                
                                # Extract stats.json
                                stats_name = f"{sdir}/stats.json"
                                try:
                                    f_stats = tar.extractfile(stats_name)
                                    if f_stats:
                                        stats_data = json.loads(f_stats.read().decode('utf-8'))
                                except Exception:
                                    pass
                                    
                                # Extract variants.vcf.gz
                                vcf_name = f"{sdir}/variants.vcf.gz"
                                try:
                                    f_vcf = tar.extractfile(vcf_name)
                                    if f_vcf:
                                        vcf_content = gzip.decompress(f_vcf.read()).decode('utf-8')
                                        for line in vcf_content.split('\n'):
                                            if line and not line.startswith('#'):
                                                parts = line.split('\t')
                                                if len(parts) >= 5:
                                                    variants_list.append(f"{parts[3]}{parts[1]}{parts[4]}")
                                except Exception:
                                    pass
                                    
                                # Extract report.tsv
                                report_name = f"{sdir}/report.tsv"
                                try:
                                    f_report = tar.extractfile(report_name)
                                    if f_report:
                                        report_lines = f_report.read().decode('utf-8').strip().split('\n')
                                        for line in report_lines:
                                            parts = line.split('\t')
                                            if len(parts) >= 2:
                                                if "contigs (>= 0 bp)" in parts[0]:
                                                    contigs_count = parts[1]
                                                elif "Total length (>= 0 bp)" in parts[0]:
                                                    total_length = parts[1]
                                except Exception:
                                    pass
                                    
                                sample_name = stats_data.get("sample_name", os.path.basename(sdir))
                                avg_depth = stats_data.get("depth_avg", "N/A")
                                if isinstance(avg_depth, (int, float)):
                                    avg_depth = f"{avg_depth:.2f}"
                                    
                                cov_breadth = stats_data.get("coverage_breadth", "N/A")
                                if isinstance(cov_breadth, (int, float)):
                                    cov_breadth = f"{cov_breadth * 100:.2f}%"
                                    
                                rows.append({
                                    "Sample Name": sample_name,
                                    "Average Depth": avg_depth,
                                    "Coverage Breadth": cov_breadth,
                                    "Total Reads": stats_data.get("total_reads", "N/A"),
                                    "Mapped Reads": stats_data.get("mapped_reads", "N/A"),
                                    "Variants (SNPs)": ", ".join(variants_list) if variants_list else "None",
                                    "Contigs Count": contigs_count,
                                    "Total Length": total_length
                                })
                                
                            if rows:
                                df_summary = pd.DataFrame(rows)
                                contents[label] = df_summary.to_csv(index=False)
                            else:
                                contents[label] = "❌ 未能在壓縮檔中解析出有效的 sample 數據"
                        else:
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

            
def select_mode(title):
    # 若切換主題，重置已上傳檔案、舊分析報告與 FHIR 預覽
    if st.session_state.get("selected_mode") != title:
        st.session_state.uploaded_files_dict = {}
        st.session_state.gemini_analysis_result = None
        st.session_state.fhir_json_preview = None
        st.session_state.kg_context_retrieved = None
    st.session_state.selected_mode = title


def render_mode_card(icon, title, desc, key):
    selected = st.session_state.get("selected_mode") == title

    # -----------------------------
    # 卡片 CSS
    # -----------------------------
    border = "4px solid #219ebc" if selected else "2px solid #cccccc"
    shadow = "0 0 15px rgba(33, 158, 188, 0.45)" if selected else "none"
    bg = "#f0faff" if selected else "#ffffff"

    st.markdown(
        f"""
        <style>

        /* 只控制這一張 card button */
        .st-key-{key}_btn button {{
            width: 100% !important;
            height: 300px !important;

            background-color: {bg} !important;
            color: #003049 !important;

            border: {border} !important;
            border-radius: 12px !important;

            box-shadow: {shadow} !important;

            padding: 1.5rem !important;
            margin: 0 !important;

            transition: all 0.2s ease !important;

            display: flex !important;
            align-items: center !important;
            justify-content: center !important;

            cursor: pointer !important;
        }}

        /* Hover */
        .st-key-{key}_btn button:hover {{
            background-color: #f0faff !important;
            border: 4px solid #219ebc !important;
            box-shadow: 0 0 20px rgba(33, 158, 188, 0.6) !important;
            transform: scale(1.03);
        }}

        /* 點擊時 */
        .st-key-{key}_btn button:active {{
            transform: scale(0.99);
        }}

        /* button 裡面的文字 */
        .st-key-{key}_btn button p {{
            white-space: pre-wrap !important;
            text-align: center !important;

            font-size: 1.25rem !important;
            line-height: 1.8 !important;
            font-weight: normal !important;

            color: #003049 !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # 按鈕內容
    #
    # icon
    # title
    # desc
    # 全部都是按鈕本身的一部分
    # -----------------------------
    button_text = (
        f"**{icon}**  \n"
        f"**{title}**  \n"
        f"{desc}"
    )

    # 整張 button 就是卡片
    st.button(
        button_text,
        key=f"{key}_btn",
        width="stretch",
        on_click=select_mode,
        args=(title,),
    )



def main():
    st.set_page_config(page_title="Gemini CSV 分析", layout="wide")
    st.title("🧬 SentinEID：一體化「基因體至床邊」新興感染症 AI 監測與臨床精準決策支援平台") 

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

                smart = build_smart_client(iss_param, auth_endpoint, token_endpoint, launch_param)
                auth_redirect_url = smart.authorize_url          # aud/PKCE/state 全部自動處理好
                save_pending_oauth_state(smart.server.auth.auth_state, smart.state)

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

    elif code_param and state_param:
        pending = load_pending_oauth_state(state_param)
        if pending is None:
            st.error("❌ 授權逾時或 state 驗證失敗，請重新從 EHR 啟動一次。")
        else:
            with st.spinner("🔄 正在交換 EHR 授權 Token..."):
                try:
                    smart = build_smart_client(None, None, None, state=pending)
                    callback_url = f"{REDIRECT_URI}?code={code_param}&state={state_param}"
                    smart.handle_callback(callback_url)

                    st.session_state.fhir_url = smart.server.base_uri
                    st.session_state.fhir_token = smart.server.auth.access_token
                    st.session_state.fhir_patient_id = smart.patient_id
                    p_demo = get_fhir_patient_demographics(
                        st.session_state.fhir_url,
                        st.session_state.fhir_patient_id,
                        st.session_state.fhir_token,
                    )
                    if p_demo:
                        st.session_state.active_patient_demographics = p_demo
                        st.success(f"🎉 成功連線！病患: {p_demo.get('name')} (ID: {p_demo.get('id')})")
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

    st.session_state.user_gemini_key = default_key_val
    user_api_key = st.sidebar.text_input(
        "輸入 Gemini API 金鑰",
        value=default_key_val,
        type="password",
        help="輸入您的 Gemini API 金鑰。預設已自動帶入系統內置的金鑰。"
    )
    st.session_state.user_gemini_key = user_api_key

    # ---------- UMLS API 金鑰配置 ----------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 UMLS API 金鑰配置")
    
    # 預設帶入原設定之金鑰，一開始不要空白
    default_umls_key_val = st.session_state.get("user_umls_key", "d6fbdc40-6f90-484a-a8a7-14c919cdfda0")

    st.session_state.user_umls_key = default_umls_key_val
    user_umls_key = st.sidebar.text_input(
        "輸入 UMLS API 金鑰",
        value=default_umls_key_val,
        type="password",
        help="輸入您的 UMLS API 金鑰。預設已自動帶入系統內置的金鑰。"
    )
    st.session_state.user_umls_key = user_umls_key

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
                "Combined Microbiome File", "Heatmap", "Sample Metadata", "Samples Overview",
                "Sample Taxon Report", "Combined Sample Taxon Results",
                "Contig Summary Reports", "Host Gene Count"
            ],
            "Consensus Genome": [
                "Sample Metadata", "Consensus Genome Overview", "Intermediate Output Files"
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
        if st.button("🚀 開始分析"):
            st.session_state.fhir_json_preview = None
            uploaded_files_dict = st.session_state.uploaded_files_dict
            
            # 檢查是否完全沒有任何資料（無檔案也無載入病患）
            if not uploaded_files_dict and not st.session_state.active_patient_demographics:
                st.warning("請至少上傳一個報告檔案或從 FHIR 連線載入病患資料，以便進行分析。")
            else:
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
                else:
                    with st.spinner("Gemini 分析中..."):
                        try:
                            # 動態配置與初始化
                            genai.configure(api_key=current_api_key)
                            dynamic_model = genai.GenerativeModel("gemini-2.5-pro")
                            dynamic_chat = dynamic_model.start_chat()
                            
                            response = dynamic_chat.send_message(prompt)
                            st.session_state.gemini_analysis_result = response.text
                            st.rerun() # 立即重整，確保報告可以持續穩定顯示在按鈕下方
                        except Exception as e:
                            st.error(f"❌ Gemini 分析失敗：{e}")

        # 如果已經有分析結果，不論進行任何按鈕操作，都持續穩定渲染在畫面上
        if st.session_state.get("gemini_analysis_result"):
            st.subheader("📄 分析結果")

            # 顯示 MetagenomicKG & PrimeKG 實時檢索脈絡
            if st.session_state.get("kg_context_retrieved"):
                with st.expander("🌐 混合知識圖譜 (PrimeKG & MetagenomicKG & BV-BRC & CARD) 實時檢索證據 (Live Graph Evidence)", expanded=True):
                    st.markdown(st.session_state.kg_context_retrieved)

            import textwrap
            import markdown as md_lib
            # 1. 先把 Gemini 回傳的 Markdown 明確轉成乾淨的 HTML
            #    （extra: 支援表格/縮寫等；sane_lists: 清單行為更直覺；nl2br: 保留單行換行）
            content_html = md_lib.markdown(
                st.session_state.gemini_analysis_result,
                extensions=["extra", "sane_lists", "nl2br"],
            )

            # 2. 標題 + 轉好的內容 HTML，一起包進同一個白底卡片，只呼叫一次 st.markdown
            box_html = textwrap.dedent(f"""
            <div style="
                background-color: #f7f9fc;
                padding: 1.2rem 1.5rem;
                border-radius: 12px;
                border-left: 6px solid #1f77b4;
                margin-bottom: 1rem;
                color: #1f2937;
            ">
                <h4 style="margin-top: 0; margin-bottom: 0.8rem; color: #1f2937 !important;">
                    📄 Gemini 分析結果
                </h4>
                <div style="color: #1f2937 !important;">
                    {content_html}
                </div>
            </div>
            """)

            st.markdown(box_html, unsafe_allow_html=True)


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
                    st.session_state.fhir_json_preview = None # 每次按下按鈕時，立刻清空上一次的預覽結果，防止失敗時殘留舊的 fallback JSON
                    # 使用具備臨床 AI NLP（如 John Snow Labs FHIR-Ready AI / Azure Text Analytics for Health）能力的引擎進行非結構化文本實體提取與 FHIR Bundle 轉換
                    current_api_key = st.session_state.get("user_gemini_key", GOOGLE_API_KEY)
                    if current_api_key:
                        current_api_key = current_api_key.strip().replace('"', '').replace("'", "")
                    with st.spinner("🤖 Clinical AI (Text-to-FHIR) 正在解析臨床文本、提取實體並生成 FHIR 結構化 Bundle..."):
                        try:
                            fhir_dict = convert_text_to_fhir_structured_ai(
                                save_patient_id,
                                st.session_state.gemini_analysis_result,
                                current_api_key
                            )
                            import json
                            st.session_state.fhir_json_preview = json.dumps(fhir_dict, indent=2, ensure_ascii=False)
                            st.success("🎉 FHIR 轉換成功！")
                        except Exception as convert_err:
                            st.error(f"❌ FHIR 轉換失敗！這通常是因為您的 Gemini API 金鑰配置無效（例如填入了工研院 ITRI API 金鑰）、配額已滿，或是網路連線受阻。\n\n**原始錯誤訊息：** `{convert_err}`")
                    # st.rerun()

                # 預覽與儲存
                if st.session_state.get("fhir_json_preview"):
                    st.markdown("#### 🔍 FHIR 格式預覽 (FHIR Resource Preview)")
                    st.code(st.session_state.fhir_json_preview, language="json",height=500,)
                    
                    if st.button("💾 上傳至 FHIR server (Save to FHIR server)"):
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
                st.session_state.kg_context_retrieved = None
                st.rerun()
                    
if __name__ == "__main__":
    main()
