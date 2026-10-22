"""
Neo4j AuraDB / 遠端集中式知識圖譜一鍵同步工具 (Sync to Remote Neo4j / AuraDB)

支援將本機 PrimeKG、MetagenomicKG、CARD 與 BV-BRC 圖譜資料一鍵同步至：
1. 官方免費 Neo4j AuraDB (neo4j+s://...databases.neo4j.io) - 供 Streamlit Cloud 使用
2. 遠端雲端伺服器 (bolt://...:7687)
3. 本地 Docker 容器 (bolt://localhost:7687)

使用方式:
  python3 sync_to_remote_neo4j.py --uri "neo4j+s://xxxx.databases.neo4j.io" --user neo4j --password "your-password"
"""

import os
import json
import argparse
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_JSON_PATH = os.path.join(_CURRENT_DIR, "offline_kg_data.json")

def sync_data(uri: str, user: str, password: str):
    logging.info(f"正在連線至目標 Neo4j 資料庫: {uri} (使用者: {user}) ...")
    
    if not os.path.exists(_JSON_PATH):
        logging.error(f"找不到圖譜資料檔案: {_JSON_PATH}")
        return False
        
    with open(_JSON_PATH, "r", encoding="utf-8") as f:
        kg_data = json.load(f)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        # 1. 建立唯一性約束與索引
        logging.info("1. 建立唯一性約束與檢索索引...")
        for label in ["Pathogen", "Disease", "Drug", "Phenotype", "AMRGene", "Strain"]:
            try:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
            except Exception as e:
                logging.warning(f"建立約束警告 ({label}): {e}")

        # 2. 匯入 Pathogen
        logging.info("2. 匯入 MetagenomicKG 微生物病原體節點...")
        for p in kg_data.get("pathogens", []):
            session.run("""
            MERGE (m:Pathogen {id: $id})
            SET m.name = $name,
                m.taxon_id = $taxon_id,
                m.gram_stain = $gram_stain,
                m.description = $desc,
                m.aliases = $aliases
            """, id=p["id"], name=p["name"], taxon_id=p.get("taxon_id", ""),
                 gram_stain=p.get("gram_stain", ""), desc=p.get("desc", ""),
                 aliases=p.get("aliases", []))

        # 3. 匯入 Disease (PrimeKG)
        logging.info("3. 匯入 PrimeKG 疾病實體節點...")
        for d in kg_data.get("diseases", []):
            session.run("""
            MERGE (dis:Disease {id: $id})
            SET dis.name = $name,
                dis.cui = $cui,
                dis.snomed_code = $snomed_code,
                dis.description = $desc,
                dis.aliases = $aliases
            """, id=d["id"], name=d["name"], cui=d.get("cui", ""),
                 snomed_code=d.get("snomed_code", ""), desc=d.get("desc", ""),
                 aliases=d.get("aliases", []))

        # 4. 匯入 Drug (PrimeKG)
        logging.info("4. 匯入 PrimeKG 處方藥物實體節點...")
        for dr in kg_data.get("drugs", []):
            session.run("""
            MERGE (drug:Drug {id: $id})
            SET drug.name = $name,
                drug.rxnorm_code = $rxnorm_code,
                drug.drug_class = $drug_class,
                drug.mechanism_of_action = $moa,
                drug.description = $desc,
                drug.aliases = $aliases
            """, id=dr["id"], name=dr["name"], rxnorm_code=dr.get("rxnorm_code", ""),
                 drug_class=dr.get("drug_class", ""), moa=dr.get("moa", ""),
                 desc=dr.get("desc", ""), aliases=dr.get("aliases", []))

        # 5. 匯入 Phenotype (PrimeKG)
        logging.info("5. 匯入 PrimeKG 臨床表型實體節點...")
        for ph in kg_data.get("phenotypes", []):
            session.run("""
            MERGE (p:Phenotype {id: $id})
            SET p.name = $name,
                p.description = $desc,
                p.aliases = $aliases
            """, id=ph["id"], name=ph["name"], desc=ph.get("desc", ""),
                 aliases=ph.get("aliases", []))

        # 6. 匯入 AMRGene (CARD)
        logging.info("6. 匯入 CARD 抗藥基因實體節點...")
        for g in kg_data.get("amr_genes", []):
            session.run("""
            MERGE (gene:AMRGene {id: $id})
            SET gene.name = $name,
                gene.drug_class = $drug_class,
                gene.clinical_significance = $clinical_sig,
                gene.aliases = $aliases
            """, id=g["id"], name=g["name"], drug_class=g.get("drug_class", ""),
                 clinical_sig=g.get("clinical_sig", ""), aliases=g.get("aliases", []))

        # 7. 匯入 Strain (BV-BRC)
        logging.info("7. 匯入 BV-BRC 病毒與細菌基因組實體節點...")
        for s in kg_data.get("strains", []):
            session.run("""
            MERGE (strain:Strain {id: $id})
            SET strain.name = $name,
                strain.scientific_name = $scientific_name,
                strain.genome_type = $genome_type,
                strain.description = $desc
            """, id=s["id"], name=s["name"], scientific_name=s.get("scientific_name", ""),
                 genome_type=s.get("genome_type", ""), desc=s.get("desc", ""))

        # 8. 建立實體關聯邊 (Relationships)
        logging.info("8. 建立多跳臨床關聯邊 (CAUSES, INDICATED_FOR, CONTRAINDICATED_FOR, HAS_PHENOTYPE)...")
        # Pathogen -> Disease (CAUSES)
        for p in kg_data.get("pathogens", []):
            for d_name in p.get("diseases", []):
                session.run("""
                MATCH (m:Pathogen {id: $pid}), (d:Disease)
                WHERE toLower(d.name) = toLower($dname)
                MERGE (m)-[:CAUSES]->(d)
                """, pid=p["id"], dname=d_name)

        # Drug -> Disease (INDICATED_FOR)
        for dr in kg_data.get("drugs", []):
            for d_name in dr.get("indications", []):
                session.run("""
                MATCH (drug:Drug {id: $did}), (d:Disease)
                WHERE toLower(d.name) = toLower($dname)
                MERGE (drug)-[:INDICATED_FOR]->(d)
                """, did=dr["id"], dname=d_name)

        # Drug -> Disease (CONTRAINDICATED_FOR)
        for dr in kg_data.get("drugs", []):
            for contra_str in dr.get("contraindications", []):
                d_target = "Severe Sepsis" if "Sepsis" in contra_str else "Pneumonia, Bacterial"
                rationale = contra_str.split("禁忌原因:")[-1].replace(")", "").strip() if "禁忌原因:" in contra_str else contra_str
                session.run("""
                MATCH (drug:Drug {id: $did}), (d:Disease)
                WHERE toLower(d.name) CONTAINS toLower($dtarget)
                MERGE (drug)-[r:CONTRAINDICATED_FOR]->(d)
                SET r.clinical_rationale = $rationale
                """, did=dr["id"], dtarget=d_target, rationale=rationale)

        # Disease -> Phenotype (HAS_PHENOTYPE)
        for d in kg_data.get("diseases", []):
            for ph_name in d.get("phenotypes", []):
                session.run("""
                MATCH (d:Disease {id: $did}), (ph:Phenotype)
                WHERE toLower(ph.name) = toLower($phname)
                MERGE (d)-[:HAS_PHENOTYPE]->(ph)
                """, did=d["id"], phname=ph_name)

        # Drug -> Phenotype (TREATS_PHENOTYPE)
        for dr in kg_data.get("drugs", []):
            for ph_name in dr.get("treats_phenotypes", []):
                session.run("""
                MATCH (drug:Drug {id: $did}), (ph:Phenotype)
                WHERE toLower(ph.name) = toLower($phname)
                MERGE (drug)-[:TREATS_PHENOTYPE]->(ph)
                """, did=dr["id"], phname=ph_name)

        # AMRGene -> Drug (CONVEYS_RESISTANCE_TO)
        for g in kg_data.get("amr_genes", []):
            for r_drug in g.get("resistant_drugs", []):
                session.run("""
                MATCH (gene:AMRGene {id: $gid}), (drug:Drug)
                WHERE toLower(drug.name) = toLower($rdrug)
                MERGE (gene)-[:CONVEYS_RESISTANCE_TO]->(drug)
                """, gid=g["id"], rdrug=r_drug)

        # AMRGene -> Pathogen (DETECTED_IN)
        for g in kg_data.get("amr_genes", []):
            for h_bact in g.get("host_bacteria", []):
                session.run("""
                MATCH (gene:AMRGene {id: $gid}), (m:Pathogen)
                WHERE toLower(m.name) = toLower($hbact)
                MERGE (gene)-[:DETECTED_IN]->(m)
                """, gid=g["id"], hbact=h_bact)

        # 統計節點與關聯
        node_cnt = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rel_cnt = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
        logging.info(f"🎉 同步完成！目標 Neo4j 目前包含 {node_cnt} 個實體節點與 {rel_cnt} 條關係邊。")
        driver.close()
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="同步知識圖譜至 Neo4j AuraDB 或遠端資料庫")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j 連線 URI (例如 neo4j+s://xxxx.databases.neo4j.io)")
    parser.add_argument("--user", default="neo4j", help="Neo4j 使用者帳號")
    parser.add_argument("--password", default="idseqneo4j", help="Neo4j 使用者密碼")
    args = parser.parse_args()

    sync_data(args.uri, args.user, args.password)
