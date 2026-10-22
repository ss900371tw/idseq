"""
Neo4j 集中式醫學與基因體知識圖譜服務 (Centralized Neo4j Medical & Genomic KG Service)
支援本機集中式 Neo4j、雲端 Neo4j AuraDB 以及 Streamlit Cloud 自動沙盒容錯。

涵蓋四大知識圖譜領域：
1. MetagenomicKG: 微生物病原體、分類階層、感染表型與抗微生物治療
2. CARD (ARO): 抗藥性基因、抗藥生化機轉與抗生素家族
3. BV-BRC: 病毒與細菌基因體學、演化譜系 (Lineages)、蛋白結構與受體網絡
4. PrimeKG: 精準醫學疾病分類 (MONDO)、臨床表型 (HPO)、適應症與用藥禁忌 (Contraindications)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, exceptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_OFFLINE_KG_PATH = os.path.join(_CURRENT_DIR, "offline_kg_data.json")

def get_default_neo4j_credentials():
    """
    依序自 Streamlit Secrets (st.secrets)、環境變數與預設值讀取 Neo4j 連線資訊，
    完美支援 Streamlit Cloud (AuraDB neo4j+s://) 與本機 Docker (bolt://localhost:7687)。
    """
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "idseqneo4j"
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            uri = st.secrets.get("NEO4J_URI", uri)
            user = st.secrets.get("NEO4J_USER", user)
            password = st.secrets.get("NEO4J_PASSWORD", password)
    except Exception:
        pass

    uri = os.getenv("NEO4J_URI", uri)
    user = os.getenv("NEO4J_USER", user)
    password = os.getenv("NEO4J_PASSWORD", password)
    return uri, user, password

DEFAULT_NEO4J_URI, DEFAULT_NEO4J_USER, DEFAULT_NEO4J_PASSWORD = get_default_neo4j_credentials()

def get_driver(uri: str = DEFAULT_NEO4J_URI, user: str = DEFAULT_NEO4J_USER, password: str = DEFAULT_NEO4J_PASSWORD):
    """建立並回傳 Neo4j Driver 實例"""
    return GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5)

def test_neo4j_connection(uri: str, user: str, password: str) -> Dict[str, Any]:
    """測試 Neo4j 連線狀態並回傳圖資料庫統計資訊"""
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=4)
        with driver.session() as session:
            version_res = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] AS version, edition")
            comp = version_res.single()
            node_cnt = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            rel_cnt = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
            driver.close()
            return {
                "success": True,
                "version": f"{comp['name']} {comp['version']} ({comp['edition']})",
                "node_count": node_cnt,
                "relationship_count": rel_cnt,
                "message": f"Neo4j 連線成功！圖庫包含 {node_cnt} 個實體節點與 {rel_cnt} 條多跳關係邊。"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Neo4j 連線失敗：{e}"
        }

# =========================================================================
# Neo4j Live Cypher 查詢函式 (本機或雲端 Neo4j AuraDB 集中管理)
# =========================================================================

def query_metagenomics_kg(session, term: str) -> List[str]:
    """查詢 MetagenomicKG：微生物病原體、致病原因、攜帶抗藥基因與有效治療藥物"""
    lines = []
    cypher = """
    MATCH (m:Pathogen)
    WHERE toLower(m.name) CONTAINS toLower($term)
       OR any(a IN m.aliases WHERE toLower(a) CONTAINS toLower($term))
       OR toLower($term) CONTAINS toLower(m.name)
    OPTIONAL MATCH (m)-[:CAUSES]->(d:Disease)
    OPTIONAL MATCH (gene:AMRGene)-[:DETECTED_IN]->(m)
    OPTIONAL MATCH (d)<-[:INDICATED_FOR]-(dr:Drug)
    RETURN m.name AS pathogen,
           m.taxon_id AS taxon_id,
           m.gram_stain AS gram_stain,
           m.description AS desc,
           collect(DISTINCT d.name) AS diseases,
           collect(DISTINCT gene.name) AS amr_genes,
           collect(DISTINCT dr.name) AS indicated_drugs
    LIMIT 3
    """
    result = session.run(cypher, term=term)
    for rec in result:
        p_name = rec["pathogen"]
        lines.append(f"📌 **MetagenomicKG 實體：{p_name}** (`{rec['taxon_id']}` | {rec['gram_stain']})")
        if rec["desc"]:
            lines.append(f"  - **病原生物學特徵**: {rec['desc']}")
        if rec["diseases"]:
            lines.append(f"  - **引起之感染症候群 (CAUSES)**: {', '.join(rec['diseases'])}")
        if rec["amr_genes"]:
            lines.append(f"  - **常見攜帶抗藥基因 (DETECTED_IN)**: {', '.join(rec['amr_genes'])}")
        if rec["indicated_drugs"]:
            lines.append(f"  - **推薦臨床治療用藥 (INDICATED_FOR)**: {', '.join(rec['indicated_drugs'])}")
        lines.append("")
    return lines

def query_amr_card_kg(session, term: str) -> List[str]:
    """查詢 CARD (ARO)：抗藥基因、抗藥生化機轉與抗生素耐藥家族"""
    lines = []
    cypher = """
    MATCH (g:AMRGene)
    WHERE toLower(g.name) CONTAINS toLower($term)
       OR any(a IN g.aliases WHERE toLower(a) CONTAINS toLower($term))
       OR toLower($term) CONTAINS toLower(g.name)
       OR toLower(g.drug_class) CONTAINS toLower($term)
    OPTIONAL MATCH (g)-[:HAS_MECHANISM]->(m:AMRMechanism)
    OPTIONAL MATCH (g)-[:CONVEYS_RESISTANCE_TO]->(dr:Drug)
    OPTIONAL MATCH (g)-[:DETECTED_IN]->(p:Pathogen)
    RETURN g.name AS gene,
           g.id AS aro_id,
           g.drug_class AS drug_class,
           g.clinical_significance AS clinical_sig,
           collect(DISTINCT m.name + ': ' + m.description) AS mechanisms,
           collect(DISTINCT dr.name) AS resistant_drugs,
           collect(DISTINCT p.name) AS host_bacteria
    LIMIT 3
    """
    result = session.run(cypher, term=term)
    for rec in result:
        g_name = rec["gene"]
        lines.append(f"📌 **CARD (ARO) 抗藥基因實體：{g_name}** (`{rec['aro_id']}` | 抵抗類別: {rec['drug_class']})")
        if rec["clinical_sig"]:
            lines.append(f"  - **臨床抗藥重要性**: {rec['clinical_sig']}")
        if rec["mechanisms"]:
            lines.append(f"  - **抗藥生化機轉 (HAS_MECHANISM)**: {'; '.join(rec['mechanisms'])}")
        if rec["resistant_drugs"]:
            lines.append(f"  - **耐藥無效藥物 (CONVEYS_RESISTANCE_TO)**: {', '.join(rec['resistant_drugs'])} ⚠️ 治療失敗高風險")
        if rec["host_bacteria"]:
            lines.append(f"  - **常見宿主菌株 (DETECTED_IN)**: {', '.join(rec['host_bacteria'])}")
        lines.append("")
    return lines

def query_consensus_genome_kg(session, term: str) -> List[str]:
    """查詢 BV-BRC：病毒/細菌變異譜系、蛋白結構、受體結合與標誌突變"""
    lines = []
    cypher = """
    MATCH (s:Strain)
    WHERE toLower(s.name) CONTAINS toLower($term)
       OR toLower(s.scientific_name) CONTAINS toLower($term)
       OR toLower($term) CONTAINS toLower(s.name)
    OPTIONAL MATCH (s)-[:HAS_LINEAGE]->(l:Lineage)
    OPTIONAL MATCH (s)-[:EXPRESSES_PROTEIN]->(p:Protein)
    OPTIONAL MATCH (p)-[:BINDS_TO]->(r:HostReceptor)
    OPTIONAL MATCH (p)-[:HAS_MUTATION]->(mut:Mutation)
    RETURN s.name AS strain,
           s.id AS strain_id,
           s.genome_type AS genome_type,
           s.description AS desc,
           collect(DISTINCT l.name + ' (' + l.description + ')') AS lineages,
           collect(DISTINCT p.name + ' [' + p.gene_symbol + ', PDB:' + p.pdb_id + ']') AS proteins,
           collect(DISTINCT r.name + ' (' + r.cellular_location + ')') AS receptors,
           collect(DISTINCT mut.name + ': ' + mut.biological_impact) AS mutations
    LIMIT 3
    """
    result = session.run(cypher, term=term)
    for rec in result:
        s_name = rec["strain"]
        lines.append(f"📌 **BV-BRC 病毒基因體學實體：{s_name}** (`{rec['strain_id']}` | 基因體類型: {rec['genome_type']})")
        if rec["desc"]:
            lines.append(f"  - **病毒演化與致病特徵**: {rec['desc']}")
        if rec["lineages"]:
            lines.append(f"  - **重點變異譜系 (HAS_LINEAGE)**: {'; '.join(rec['lineages'])}")
        if rec["proteins"]:
            lines.append(f"  - **關鍵結構/功能蛋白 (EXPRESSES_PROTEIN)**: {', '.join(rec['proteins'])}")
        if rec["receptors"]:
            lines.append(f"  - **宿主結合受體 (BINDS_TO)**: {', '.join(rec['receptors'])}")
        if rec["mutations"]:
            lines.append(f"  - **核心突變位點特徵 (HAS_MUTATION)**: {'; '.join(rec['mutations'])}")
        lines.append("")
    return lines

def query_primekg_clinical_kg(session, term: str) -> List[str]:
    """
    查詢 PrimeKG 精準醫學圖譜：
    1. 疾病實體 (Disease / MONDO): 關聯臨床表型 (HPO)、常見感染致病原、適應症藥物與關鍵用藥禁忌
    2. 處方用藥實體 (Drug / DrugBank / RxNorm): 藥理機轉、適應疾病、用藥禁忌症、對症表型、已知抗藥基因
    3. 臨床表型實體 (Phenotype / HPO): 表型特徵、相關聯疾病、對症緩解藥物
    """
    lines = []
    clean_term = term.strip()
    if not clean_term:
        return lines

    # A. 疾病節點查詢
    cypher_disease = """
    MATCH (d:Disease)
    WHERE toLower(d.name) CONTAINS toLower($term)
       OR any(a IN d.aliases WHERE toLower(a) CONTAINS toLower($term))
       OR toLower($term) CONTAINS toLower(d.name)
    OPTIONAL MATCH (d)-[:HAS_PHENOTYPE]->(ph:Phenotype)
    OPTIONAL MATCH (d)<-[:INDICATED_FOR]-(dr:Drug)
    OPTIONAL MATCH (d)<-[contra:CONTRAINDICATED_FOR]-(bad_dr:Drug)
    OPTIONAL MATCH (p:Pathogen)-[:CAUSES]->(d)
    RETURN d.name AS disease,
           d.id AS mondo_id,
           d.cui AS cui,
           d.snomed_code AS snomed_code,
           d.description AS desc,
           collect(DISTINCT ph.name) AS phenotypes,
           collect(DISTINCT dr.name) AS indicated_drugs,
           collect(DISTINCT bad_dr.name + ' (禁忌原因: ' + contra.clinical_rationale + ')') AS contraindications,
           collect(DISTINCT p.name) AS causative_pathogens
    LIMIT 3
    """
    res_dis = list(session.run(cypher_disease, term=clean_term))
    for rec in res_dis:
        d_name = rec["disease"]
        lines.append(f"📌 **PrimeKG 精準醫學疾病實體：{d_name}** (`{rec['mondo_id']}` | SNOMED CT: `{rec['snomed_code']}` | UMLS CUI: `{rec['cui']}`)")
        if rec["desc"]:
            lines.append(f"  - **病理機轉與定義**: {rec['desc']}")
        if rec["phenotypes"]:
            lines.append(f"  - **關聯臨床表型 (HAS_PHENOTYPE)**: {', '.join(rec['phenotypes'])}")
        if rec["causative_pathogens"]:
            lines.append(f"  - **常見感染致病原 (CAUSES)**: {', '.join(rec['causative_pathogens'])}")
        if rec["indicated_drugs"]:
            lines.append(f"  - **臨床推薦適應藥物 (INDICATED_FOR)**: {', '.join(rec['indicated_drugs'])}")
        if rec["contraindications"]:
            lines.append("  - **⚠️ 嚴格用藥禁忌警示 (CONTRAINDICATED_FOR)**:")
            for contra in rec["contraindications"]:
                lines.append(f"    * {contra}")
        lines.append("")

    # B. 處方用藥節點查詢
    cypher_drug = """
    MATCH (dr:Drug)
    WHERE toLower(dr.name) CONTAINS toLower($term)
       OR any(a IN dr.aliases WHERE toLower(a) CONTAINS toLower($term))
       OR toLower($term) CONTAINS toLower(dr.name)
       OR toLower(dr.drug_class) CONTAINS toLower($term)
    OPTIONAL MATCH (dr)-[:INDICATED_FOR]->(d:Disease)
    OPTIONAL MATCH (dr)-[contra:CONTRAINDICATED_FOR]->(bad_d:Disease)
    OPTIONAL MATCH (dr)-[:TREATS_PHENOTYPE]->(ph:Phenotype)
    OPTIONAL MATCH (g:AMRGene)-[:CONVEYS_RESISTANCE_TO]->(dr)
    RETURN dr.name AS drug,
           dr.id AS drugbank_id,
           dr.rxnorm_code AS rxnorm_code,
           dr.drug_class AS drug_class,
           dr.mechanism_of_action AS moa,
           dr.description AS desc,
           collect(DISTINCT d.name) AS indications,
           collect(DISTINCT bad_d.name + ' (禁忌原因: ' + contra.clinical_rationale + ')') AS contraindications,
           collect(DISTINCT ph.name) AS treats_phenotypes,
           collect(DISTINCT g.name) AS resistant_genes
    LIMIT 3
    """
    res_drug = list(session.run(cypher_drug, term=clean_term))
    for rec in res_drug:
        dr_name = rec["drug"]
        lines.append(f"📌 **PrimeKG 精準處方藥物實體：{dr_name}** (`{rec['drugbank_id']}` | RxNorm: `{rec['rxnorm_code']}` | 藥物類別: {rec['drug_class']})")
        if rec["moa"]:
            lines.append(f"  - **藥理生化機轉 (MoA)**: {rec['moa']}")
        if rec["desc"]:
            lines.append(f"  - **臨床藥物說明**: {rec['desc']}")
        if rec["indications"]:
            lines.append(f"  - **適應症疾病 (INDICATED_FOR)**: {', '.join(rec['indications'])}")
        if rec["contraindications"]:
            lines.append("  - **⚠️ 嚴格臨床用藥禁忌警示 (CONTRAINDICATED_FOR)**:")
            for contra in rec["contraindications"]:
                lines.append(f"    * {contra}")
        if rec["treats_phenotypes"]:
            lines.append(f"  - **對症緩解臨床表型 (TREATS_PHENOTYPE)**: {', '.join(rec['treats_phenotypes'])}")
        if rec["resistant_genes"]:
            lines.append(f"  - **已知耐藥突變/基因 (CONVEYS_RESISTANCE_TO)**: {', '.join(rec['resistant_genes'])} ⚠️ 治療失效警示")
        lines.append("")

    # C. 臨床表型節點查詢
    cypher_phenotype = """
    MATCH (ph:Phenotype)
    WHERE toLower(ph.name) CONTAINS toLower($term)
       OR any(a IN ph.aliases WHERE toLower(a) CONTAINS toLower($term))
       OR toLower($term) CONTAINS toLower(ph.name)
    OPTIONAL MATCH (d:Disease)-[:HAS_PHENOTYPE]->(ph)
    OPTIONAL MATCH (dr:Drug)-[:TREATS_PHENOTYPE]->(ph)
    RETURN ph.name AS phenotype,
           ph.id AS hpo_id,
           ph.description AS desc,
           collect(DISTINCT d.name) AS associated_diseases,
           collect(DISTINCT dr.name) AS symptomatic_drugs
    LIMIT 3
    """
    res_ph = list(session.run(cypher_phenotype, term=clean_term))
    for rec in res_ph:
        ph_name = rec["phenotype"]
        lines.append(f"📌 **PrimeKG 臨床表型實體：{ph_name}** (`{rec['hpo_id']}`)")
        if rec["desc"]:
            lines.append(f"  - **臨床表型定義**: {rec['desc']}")
        if rec["associated_diseases"]:
            lines.append(f"  - **相關聯重大疾病 (HAS_PHENOTYPE)**: {', '.join(rec['associated_diseases'])}")
        if rec["symptomatic_drugs"]:
            lines.append(f"  - **對症緩解處方推薦 (TREATS_PHENOTYPE)**: {', '.join(rec['symptomatic_drugs'])}")
        lines.append("")

    return lines

# =========================================================================
# Streamlit Cloud 離線沙盒展示圖譜備用引擎 (Cloud Sandbox Fallback Engine)
# =========================================================================

def _load_cloud_sandbox_data() -> Dict[str, Any]:
    if os.path.exists(_OFFLINE_KG_PATH):
        try:
            with open(_OFFLINE_KG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"讀取沙盒圖譜資料失敗: {e}")
    return {}

def _term_match_cloud(term: str, name: str, aliases: Optional[List[str]] = None, extra: Optional[str] = None) -> bool:
    t = term.lower().strip()
    if not t: return False
    n = name.lower()
    if t in n or n in t: return True
    if aliases:
        for a in aliases:
            al = a.lower()
            if t in al or al in t: return True
    if extra:
        ex = extra.lower()
        if t in ex or ex in t: return True
    return False

def retrieve_cloud_sandbox_kg(file_terms: List[str], records_terms: List[str], mode: str, uri: str) -> str:
    """
    當運行於 Streamlit Cloud 且未指定或尚未連線至外部 Neo4j AuraDB 時，
    自動無縫啟用沙盒展示圖譜，確保 100% 演示可用且不跳出 Connection Refused 報錯。
    """
    data = _load_cloud_sandbox_data()
    sections = [
        "🌐 **[Neo4j 集中式知識圖譜檢索 (Streamlit Cloud 沙盒展示模式)]**",
        f"• 運作說明: 當前處於雲端展示模式 (未偵測到本地/遠端 Neo4j 實例 `{uri}`)，系統已自動啟用內置 Neo4j 沙盒實證圖譜，保障完整臨床 GraphRAG 分析機能。",
        f"• 圖譜規模: 48 個核心臨床實體節點 | 117 條多跳關係邊",
        f"• 整合體系: PrimeKG (精準醫學) + MetagenomicKG (宏基因組) + CARD (抗藥性) + BV-BRC (病毒基因組)",
        f"• 查詢分析主題 (Mode): `{mode}`",
        f"• 上傳檔案實體 (File Entities): {', '.join(file_terms) if file_terms else '無'}",
        f"• 病患臨床病歷實體 (Patient Entities): {', '.join(records_terms) if records_terms else '無'}\n"
    ]

    # 1. 檔案實體
    if file_terms:
        sections.append(f"### 🧬 上傳檔案生物體與基因組圖譜實體檢索 (Mode: {mode})")
        for term in file_terms:
            matched_lines = []
            if mode == "Metagenomics":
                for p in data.get("pathogens", []):
                    if _term_match_cloud(term, p["name"], p.get("aliases")):
                        matched_lines.append(f"📌 **MetagenomicKG 實體：{p['name']}** (`{p.get('taxon_id', '')}` | {p.get('gram_stain', '')})")
                        if p.get("desc"): matched_lines.append(f"  - **病原生物學特徵**: {p['desc']}")
                        if p.get("diseases"): matched_lines.append(f"  - **引起之感染症候群 (CAUSES)**: {', '.join(p['diseases'])}")
                        if p.get("amr_genes"): matched_lines.append(f"  - **常見攜帶抗藥基因 (DETECTED_IN)**: {', '.join(p['amr_genes'])}")
                        if p.get("indicated_drugs"): matched_lines.append(f"  - **推薦臨床治療用藥 (INDICATED_FOR)**: {', '.join(p['indicated_drugs'])}")
                        matched_lines.append("")
            elif mode == "Consensus Genome":
                for s in data.get("strains", []):
                    if _term_match_cloud(term, s["name"], extra=s.get("scientific_name")):
                        matched_lines.append(f"📌 **BV-BRC 病毒基因體學實體：{s['name']}** (`{s.get('id', '')}` | 基因體類型: {s.get('genome_type', '')})")
                        if s.get("desc"): matched_lines.append(f"  - **病毒演化與致病特徵**: {s['desc']}")
                        if s.get("lineages"): matched_lines.append(f"  - **重點變異譜系 (HAS_LINEAGE)**: {'; '.join(s['lineages'])}")
                        if s.get("proteins"): matched_lines.append(f"  - **關鍵結構/功能蛋白 (EXPRESSES_PROTEIN)**: {', '.join(s['proteins'])}")
                        if s.get("receptors"): matched_lines.append(f"  - **宿主結合受體 (BINDS_TO)**: {', '.join(s['receptors'])}")
                        if s.get("mutations"): matched_lines.append(f"  - **核心突變位點特徵 (HAS_MUTATION)**: {'; '.join(s['mutations'])}")
                        matched_lines.append("")
            elif mode == "Antimicrobial Resistance":
                for g in data.get("amr_genes", []):
                    if _term_match_cloud(term, g["name"], g.get("aliases"), g.get("drug_class")):
                        matched_lines.append(f"📌 **CARD (ARO) 抗藥基因實體：{g['name']}** (`{g.get('id', '')}` | 抵抗類別: {g.get('drug_class', '')})")
                        if g.get("clinical_sig"): matched_lines.append(f"  - **臨床抗藥重要性**: {g['clinical_sig']}")
                        if g.get("mechanisms"): matched_lines.append(f"  - **抗藥生化機轉 (HAS_MECHANISM)**: {'; '.join(g['mechanisms'])}")
                        if g.get("resistant_drugs"): matched_lines.append(f"  - **耐藥無效藥物 (CONVEYS_RESISTANCE_TO)**: {', '.join(g['resistant_drugs'])} ⚠️ 治療失敗高風險")
                        if g.get("host_bacteria"): matched_lines.append(f"  - **常見宿主菌株 (DETECTED_IN)**: {', '.join(g['host_bacteria'])}")
                        matched_lines.append("")

            if matched_lines:
                sections.extend(matched_lines)

    # 2. 病患臨床病歷 (PrimeKG)
    if records_terms:
        sections.append("### 📋 病患臨床病歷精準醫學網絡檢索 (PrimeKG Clinical Context)")
        for term in records_terms:
            clinical_lines = []
            for d in data.get("diseases", []):
                if _term_match_cloud(term, d["name"], d.get("aliases")):
                    clinical_lines.append(f"📌 **PrimeKG 精準醫學疾病實體：{d['name']}** (`{d['id']}` | SNOMED CT: `{d.get('snomed_code', '')}` | UMLS CUI: `{d.get('cui', '')}`)")
                    if d.get("desc"): clinical_lines.append(f"  - **病理機轉與定義**: {d['desc']}")
                    if d.get("phenotypes"): clinical_lines.append(f"  - **關聯臨床表型 (HAS_PHENOTYPE)**: {', '.join(d['phenotypes'])}")
                    if d.get("causative_pathogens"): clinical_lines.append(f"  - **常見感染致病原 (CAUSES)**: {', '.join(d['causative_pathogens'])}")
                    if d.get("indicated_drugs"): clinical_lines.append(f"  - **臨床推薦適應藥物 (INDICATED_FOR)**: {', '.join(d['indicated_drugs'])}")
                    if d.get("contraindications"):
                        clinical_lines.append("  - **⚠️ 嚴格用藥禁忌警示 (CONTRAINDICATED_FOR)**:")
                        for contra in d["contraindications"]: clinical_lines.append(f"    * {contra}")
                    clinical_lines.append("")
            for dr in data.get("drugs", []):
                if _term_match_cloud(term, dr["name"], dr.get("aliases"), dr.get("drug_class")):
                    clinical_lines.append(f"📌 **PrimeKG 精準處方藥物實體：{dr['name']}** (`{dr['id']}` | RxNorm: `{dr.get('rxnorm_code', '')}` | 藥物類別: {dr.get('drug_class', '')})")
                    if dr.get("moa"): clinical_lines.append(f"  - **藥理生化機轉 (MoA)**: {dr['moa']}")
                    if dr.get("desc"): clinical_lines.append(f"  - **臨床藥物說明**: {dr['desc']}")
                    if dr.get("indications"): clinical_lines.append(f"  - **適應症疾病 (INDICATED_FOR)**: {', '.join(dr['indications'])}")
                    if dr.get("contraindications"):
                        clinical_lines.append("  - **⚠️ 嚴格臨床用藥禁忌警示 (CONTRAINDICATED_FOR)**:")
                        for contra in dr["contraindications"]: clinical_lines.append(f"    * {contra}")
                    if dr.get("treats_phenotypes"): clinical_lines.append(f"  - **對症緩解臨床表型 (TREATS_PHENOTYPE)**: {', '.join(dr['treats_phenotypes'])}")
                    if dr.get("resistant_genes"): clinical_lines.append(f"  - **已知耐藥突變/基因 (CONVEYS_RESISTANCE_TO)**: {', '.join(dr['resistant_genes'])} ⚠️ 治療失效警示")
                    clinical_lines.append("")
            for ph in data.get("phenotypes", []):
                if _term_match_cloud(term, ph["name"], ph.get("aliases")):
                    clinical_lines.append(f"📌 **PrimeKG 臨床表型實體：{ph['name']}** (`{ph['id']}`)")
                    if ph.get("desc"): clinical_lines.append(f"  - **臨床表型定義**: {ph['desc']}")
                    if ph.get("associated_diseases"): clinical_lines.append(f"  - **相關聯重大疾病 (HAS_PHENOTYPE)**: {', '.join(ph['associated_diseases'])}")
                    if ph.get("symptomatic_drugs"): clinical_lines.append(f"  - **對症緩解處方推薦 (TREATS_PHENOTYPE)**: {', '.join(ph['symptomatic_drugs'])}")
                    clinical_lines.append("")

            if clinical_lines:
                sections.extend(clinical_lines)

    return "\n".join(sections)

# =========================================================================
# 集中式 Neo4j 主檢索入口 (支援 Live Cypher 與 Streamlit Cloud 沙盒無縫自動切換)
# =========================================================================

def retrieve_centralized_neo4j_kg(
    file_terms: List[str],
    records_terms: List[str],
    mode: str,
    uri: str = DEFAULT_NEO4J_URI,
    user: str = DEFAULT_NEO4J_USER,
    password: str = DEFAULT_NEO4J_PASSWORD
) -> str:
    """
    執行集中式 Neo4j 圖資料庫檢索：
    1. 優先嘗試連線至指定的集中式 Neo4j 伺服器 (本機 Docker 或遠端 Neo4j AuraDB)。
    2. 若處於 Streamlit Cloud 且尚未配置外部 AuraDB 時，自動啟用內置沙盒圖譜，保證 100% 演示可用且不中斷。
    """
    sections = []
    
    # 建立連接測試 (超時時間 3 秒)
    conn_info = test_neo4j_connection(uri, user, password)
    if not conn_info["success"]:
        logging.info(f"Neo4j 未連線 ({uri})，自動轉入 Streamlit Cloud 沙盒展示圖譜模式")
        return retrieve_cloud_sandbox_kg(file_terms, records_terms, mode, uri)

    sections.append(
        f"🌐 **[Neo4j 集中式知識圖譜即時 Cypher 檢索 (Centralized Live Cypher)]**\n"
        f"• Neo4j 伺服器: `{uri}` | 核心節點數: {conn_info['node_count']} | 關聯邊數: {conn_info['relationship_count']}\n"
        f"• 查詢分析主題 (Mode): `{mode}`\n"
        f"• 上傳檔案實體 (File Entities): {', '.join(file_terms) if file_terms else '無'}\n"
        f"• 病患臨床病歷實體 (Patient Entities): {', '.join(records_terms) if records_terms else '無'}\n"
    )

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=4)
        with driver.session() as session:
            # 1. 針對上傳檔案中的生物資訊/基因體實體檢索
            if file_terms:
                sections.append(f"### 🧬 上傳檔案生物體與基因組圖譜實體檢索 (Mode: {mode})")
                for term in file_terms:
                    matched_lines = []
                    if mode == "Metagenomics":
                        matched_lines = query_metagenomics_kg(session, term)
                    elif mode == "Consensus Genome":
                        matched_lines = query_consensus_genome_kg(session, term)
                        if not matched_lines:
                            matched_lines = query_metagenomics_kg(session, term)
                    elif mode == "Antimicrobial Resistance":
                        matched_lines = query_amr_card_kg(session, term)
                        if not matched_lines:
                            matched_lines = query_metagenomics_kg(session, term)
                    
                    if matched_lines:
                        sections.extend(matched_lines)
                    else:
                        general_lines = query_metagenomics_kg(session, term) or query_amr_card_kg(session, term)
                        if general_lines:
                            sections.extend(general_lines)

            # 2. 針對病患臨床病歷 (FHIR Clinical Records) 進行 PrimeKG 與對應網絡檢索
            if records_terms:
                sections.append("### 📋 病患臨床病歷精準醫學網絡檢索 (PrimeKG Clinical Context)")
                for term in records_terms:
                    clinical_lines = query_primekg_clinical_kg(session, term)
                    if clinical_lines:
                        sections.extend(clinical_lines)
                    else:
                        alt_lines = query_amr_card_kg(session, term) or query_metagenomics_kg(session, term)
                        if alt_lines:
                            sections.extend(alt_lines)

        driver.close()
    except Exception as e:
        logging.warning(f"Neo4j 查詢執行期間發生例外 ({e})，自動回退至沙盒展示圖譜")
        return retrieve_cloud_sandbox_kg(file_terms, records_terms, mode, uri)

    return "\n".join(sections)
