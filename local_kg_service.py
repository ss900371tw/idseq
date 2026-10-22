"""
SentinEID 臨床精準醫學與基因體知識圖譜服務 (Knowledge Graph Service)
支援標準 HTTP REST API 與 本地內置引擎雙軌架構 (REST API + Local Embedded Fallback)

1. REST API 優先模式：透過標準 HTTP POST `/api/kg/query` 請求微服務端點
2. 本地內置容錯模式：若 REST API 未啟動或網路不可達，自動無縫啟用本機記憶體圖譜引擎 (Zero-Network Fallback)
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_OFFLINE_KG_PATH = os.path.join(_CURRENT_DIR, "offline_kg_data.json")
DEFAULT_KG_REST_URL = os.getenv("KG_REST_API_URL", "http://localhost:8088")

class LocalKnowledgeGraph:
    """純本地端記憶體知識圖譜引擎（提供 REST API 伺服器後端與離線備用檢索）"""

    def __init__(self, data_path: str = _OFFLINE_KG_PATH):
        self.data_path = data_path
        self.data: Dict[str, Any] = self._load_data()
        self.stats = self._calculate_stats()

    def _load_data(self) -> Dict[str, Any]:
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"讀取本地知識圖譜檔案失敗: {e}")
        return {
            "pathogens": [],
            "diseases": [],
            "drugs": [],
            "phenotypes": [],
            "amr_genes": [],
            "strains": []
        }

    def _calculate_stats(self) -> Dict[str, Any]:
        p_cnt = len(self.data.get("pathogens", []))
        d_cnt = len(self.data.get("diseases", []))
        dr_cnt = len(self.data.get("drugs", []))
        ph_cnt = len(self.data.get("phenotypes", []))
        amr_cnt = len(self.data.get("amr_genes", []))
        s_cnt = len(self.data.get("strains", []))
        total_nodes = p_cnt + d_cnt + dr_cnt + ph_cnt + amr_cnt + s_cnt

        rel_cnt = 0
        for p in self.data.get("pathogens", []):
            rel_cnt += len(p.get("diseases", [])) + len(p.get("amr_genes", [])) + len(p.get("indicated_drugs", []))
        for d in self.data.get("diseases", []):
            rel_cnt += len(d.get("phenotypes", [])) + len(d.get("indicated_drugs", [])) + len(d.get("contraindications", []))
        for dr in self.data.get("drugs", []):
            rel_cnt += len(dr.get("indications", [])) + len(dr.get("contraindications", [])) + len(dr.get("treats_phenotypes", [])) + len(dr.get("resistant_genes", []))
        for g in self.data.get("amr_genes", []):
            rel_cnt += len(g.get("mechanisms", [])) + len(g.get("resistant_drugs", [])) + len(g.get("host_bacteria", []))
        for s in self.data.get("strains", []):
            rel_cnt += len(s.get("lineages", [])) + len(s.get("proteins", [])) + len(s.get("receptors", [])) + len(s.get("mutations", []))

        return {
            "node_count": total_nodes,
            "relationship_count": rel_cnt,
            "details": {
                "pathogens": p_cnt,
                "diseases": d_cnt,
                "drugs": dr_cnt,
                "phenotypes": ph_cnt,
                "amr_genes": amr_cnt,
                "strains": s_cnt
            }
        }

    def term_matches(self, term: str, name: str, aliases: Optional[List[str]] = None, extra: Optional[str] = None) -> bool:
        t = term.lower().strip()
        if not t:
            return False
        n = name.lower()
        if t in n or n in t:
            return True
        if aliases:
            for a in aliases:
                al = a.lower()
                if t in al or al in t:
                    return True
        if extra:
            ex = extra.lower()
            if t in ex or ex in t:
                return True
        return False

    def query_metagenomics(self, term: str) -> List[str]:
        lines = []
        for p in self.data.get("pathogens", []):
            if self.term_matches(term, p["name"], p.get("aliases")):
                lines.append(f"📌 **MetagenomicKG 實體：{p['name']}** (`{p.get('taxon_id', '')}` | {p.get('gram_stain', '')})")
                if p.get("desc"): lines.append(f"  - **病原生物學特徵**: {p['desc']}")
                if p.get("diseases"): lines.append(f"  - **引起之感染症候群 (CAUSES)**: {', '.join(p['diseases'])}")
                if p.get("amr_genes"): lines.append(f"  - **常見攜帶抗藥基因 (DETECTED_IN)**: {', '.join(p['amr_genes'])}")
                if p.get("indicated_drugs"): lines.append(f"  - **推薦臨床治療用藥 (INDICATED_FOR)**: {', '.join(p['indicated_drugs'])}")
                lines.append("")
        return lines

    def query_amr_card(self, term: str) -> List[str]:
        lines = []
        for g in self.data.get("amr_genes", []):
            if self.term_matches(term, g["name"], g.get("aliases"), g.get("drug_class")):
                lines.append(f"📌 **CARD (ARO) 抗藥基因實體：{g['name']}** (`{g.get('id', '')}` | 抵抗類別: {g.get('drug_class', '')})")
                if g.get("clinical_sig"): lines.append(f"  - **臨床抗藥重要性**: {g['clinical_sig']}")
                if g.get("mechanisms"): lines.append(f"  - **抗藥生化機轉 (HAS_MECHANISM)**: {'; '.join(g['mechanisms'])}")
                if g.get("resistant_drugs"): lines.append(f"  - **耐藥無效藥物 (CONVEYS_RESISTANCE_TO)**: {', '.join(g['resistant_drugs'])} ⚠️ 治療失敗高風險")
                if g.get("host_bacteria"): lines.append(f"  - **常見宿主菌株 (DETECTED_IN)**: {', '.join(g['host_bacteria'])}")
                lines.append("")
        return lines

    def query_consensus_genome(self, term: str) -> List[str]:
        lines = []
        for s in self.data.get("strains", []):
            if self.term_matches(term, s["name"], extra=s.get("scientific_name")):
                lines.append(f"📌 **BV-BRC 病毒基因體學實體：{s['name']}** (`{s.get('id', '')}` | 基因體類型: {s.get('genome_type', '')})")
                if s.get("desc"): lines.append(f"  - **病毒演化與致病特徵**: {s['desc']}")
                if s.get("lineages"): lines.append(f"  - **重點變異譜系 (HAS_LINEAGE)**: {'; '.join(s['lineages'])}")
                if s.get("proteins"): lines.append(f"  - **關鍵結構/功能蛋白 (EXPRESSES_PROTEIN)**: {', '.join(s['proteins'])}")
                if s.get("receptors"): lines.append(f"  - **宿主結合受體 (BINDS_TO)**: {', '.join(s['receptors'])}")
                if s.get("mutations"): lines.append(f"  - **核心突變位點特徵 (HAS_MUTATION)**: {'; '.join(s['mutations'])}")
                lines.append("")
        return lines

    def query_primekg(self, term: str) -> List[str]:
        lines = []
        # 1. 疾病 (Disease)
        for d in self.data.get("diseases", []):
            if self.term_matches(term, d["name"], d.get("aliases")):
                lines.append(f"📌 **PrimeKG 精準醫學疾病實體：{d['name']}** (`{d['id']}` | SNOMED CT: `{d.get('snomed_code', '')}` | UMLS CUI: `{d.get('cui', '')}`)")
                if d.get("desc"): lines.append(f"  - **病理機轉與定義**: {d['desc']}")
                if d.get("phenotypes"): lines.append(f"  - **關聯臨床表型 (HAS_PHENOTYPE)**: {', '.join(d['phenotypes'])}")
                if d.get("causative_pathogens"): lines.append(f"  - **常見感染致病原 (CAUSES)**: {', '.join(d['causative_pathogens'])}")
                if d.get("indicated_drugs"): lines.append(f"  - **臨床推薦適應藥物 (INDICATED_FOR)**: {', '.join(d['indicated_drugs'])}")
                if d.get("contraindications"):
                    lines.append("  - **⚠️ 嚴格用藥禁忌警示 (CONTRAINDICATED_FOR)**:")
                    for contra in d["contraindications"]:
                        lines.append(f"    * {contra}")
                lines.append("")

        # 2. 處方藥物 (Drug)
        for dr in self.data.get("drugs", []):
            if self.term_matches(term, dr["name"], dr.get("aliases"), dr.get("drug_class")):
                lines.append(f"📌 **PrimeKG 精準處方藥物實體：{dr['name']}** (`{dr['id']}` | RxNorm: `{dr.get('rxnorm_code', '')}` | 藥物類別: {dr.get('drug_class', '')})")
                if dr.get("moa"): lines.append(f"  - **藥理生化機轉 (MoA)**: {dr['moa']}")
                if dr.get("desc"): lines.append(f"  - **臨床藥物說明**: {dr['desc']}")
                if dr.get("indications"): lines.append(f"  - **適應症疾病 (INDICATED_FOR)**: {', '.join(dr['indications'])}")
                if dr.get("contraindications"):
                    lines.append("  - **⚠️ 嚴格臨床用藥禁忌警示 (CONTRAINDICATED_FOR)**:")
                    for contra in dr["contraindications"]:
                        lines.append(f"    * {contra}")
                if dr.get("treats_phenotypes"): lines.append(f"  - **對症緩解臨床表型 (TREATS_PHENOTYPE)**: {', '.join(dr['treats_phenotypes'])}")
                if dr.get("resistant_genes"): lines.append(f"  - **已知耐藥突變/基因 (CONVEYS_RESISTANCE_TO)**: {', '.join(dr['resistant_genes'])} ⚠️ 治療失效警示")
                lines.append("")

        # 3. 臨床表型 (Phenotype)
        for ph in self.data.get("phenotypes", []):
            if self.term_matches(term, ph["name"], ph.get("aliases")):
                lines.append(f"📌 **PrimeKG 臨床表型實體：{ph['name']}** (`{ph['id']}`)")
                if ph.get("desc"): lines.append(f"  - **臨床表型定義**: {ph['desc']}")
                if ph.get("associated_diseases"): lines.append(f"  - **相關聯重大疾病 (HAS_PHENOTYPE)**: {', '.join(ph['associated_diseases'])}")
                if ph.get("symptomatic_drugs"): lines.append(f"  - **對症緩解處方推薦 (TREATS_PHENOTYPE)**: {', '.join(ph['symptomatic_drugs'])}")
                lines.append("")

        return lines

# 全域單例實例
_LOCAL_KG_INSTANCE: Optional[LocalKnowledgeGraph] = None

def get_local_kg() -> LocalKnowledgeGraph:
    global _LOCAL_KG_INSTANCE
    if _LOCAL_KG_INSTANCE is None:
        _LOCAL_KG_INSTANCE = LocalKnowledgeGraph()
    return _LOCAL_KG_INSTANCE

def test_kg_rest_api(api_url: str = DEFAULT_KG_REST_URL) -> Dict[str, Any]:
    """檢驗知識圖譜 REST API 伺服器連線狀態"""
    clean_url = api_url.rstrip("/")
    try:
        r = requests.get(f"{clean_url}/health", timeout=2)
        if r.status_code == 200:
            stats_r = requests.get(f"{clean_url}/api/kg/stats", timeout=2)
            stats_data = stats_r.json().get("stats", {}) if stats_r.status_code == 200 else {}
            node_cnt = stats_data.get("node_count", 48)
            rel_cnt = stats_data.get("relationship_count", 117)
            return {
                "success": True,
                "url": clean_url,
                "node_count": node_cnt,
                "relationship_count": rel_cnt,
                "message": f"REST API 連線正常！伺服器運作中 (包含 {node_cnt} 實體節點與 {rel_cnt} 關係邊)。"
            }
    except Exception as e:
        pass
    return {
        "success": False,
        "url": clean_url,
        "message": f"REST API ({clean_url}) 尚未連線（系統將自動使用本地內建圖譜引擎作為備用，保證分析正常）。"
    }

def retrieve_local_kg(
    file_terms: List[str],
    records_terms: List[str],
    mode: str,
    rest_url: Optional[str] = None
) -> str:
    """
    透過標準 HTTP REST API (預設 http://localhost:8000) 進行知識圖譜檢索。
    若 REST API 伺服器未啟動，自動啟用內置本地圖譜引擎確保零中斷。
    """
    target_api = (rest_url or os.getenv("KG_REST_API_URL", DEFAULT_KG_REST_URL)).rstrip("/")
    try:
        payload = {
            "file_terms": file_terms,
            "records_terms": records_terms,
            "mode": mode
        }
        resp = requests.post(f"{target_api}/api/kg/query", json=payload, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("markdown_context", "")
    except Exception as e:
        logging.info(f"REST API ({target_api}) 連線未回應，使用本地內置引擎進行圖譜檢索: {e}")

    # 本地內置引擎備用處理
    kg = get_local_kg()
    stats = kg.stats

    sections = [
        "🌐 **[本地內置精準醫學知識圖譜檢索 (Local Embedded KG Engine)]**",
        f"• 運作模式: 純本地端零網路依賴 (Zero-Network Dependency | In-Memory Deterministic Graph)",
        f"• 圖譜規模: {stats['node_count']} 個臨床核心節點 | {stats['relationship_count']} 條多跳實證關係邊",
        f"• 知識涵蓋: PrimeKG (精準醫學) + MetagenomicKG (宏基因組) + CARD (抗藥性) + BV-BRC (病毒基因組)",
        f"• 查詢分析主題 (Mode): `{mode}`",
        f"• 上傳檔案實體 (File Entities): {', '.join(file_terms) if file_terms else '無'}",
        f"• 病患臨床病歷實體 (Patient Entities): {', '.join(records_terms) if records_terms else '無'}\n"
    ]

    # 1. 上傳檔案生物/基因組實體檢索
    if file_terms:
        sections.append(f"### 🧬 上傳檔案生物體與基因組圖譜實體檢索 (Mode: {mode})")
        for term in file_terms:
            matched_lines = []
            if mode == "Metagenomics":
                matched_lines = kg.query_metagenomics(term)
            elif mode == "Consensus Genome":
                matched_lines = kg.query_consensus_genome(term) or kg.query_metagenomics(term)
            elif mode == "Antimicrobial Resistance":
                matched_lines = kg.query_amr_card(term) or kg.query_metagenomics(term)

            if matched_lines:
                sections.extend(matched_lines)
            else:
                fallback_lines = kg.query_metagenomics(term) or kg.query_amr_card(term)
                if fallback_lines:
                    sections.extend(fallback_lines)

    # 2. 病患臨床病歷精準醫學網絡檢索 (PrimeKG)
    if records_terms:
        sections.append("### 📋 病患臨床病歷精準醫學網絡檢索 (PrimeKG Clinical Context)")
        for term in records_terms:
            clinical_lines = kg.query_primekg(term)
            if clinical_lines:
                sections.extend(clinical_lines)
            else:
                alt_lines = kg.query_amr_card(term) or kg.query_metagenomics(term)
                if alt_lines:
                    sections.extend(alt_lines)

    return "\n".join(sections)
