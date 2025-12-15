import asyncio
from dataclasses import dataclass
from typing import Optional
from dataclasses import asdict
import numpy as np
from retrieval import DB_CONFIG, TABLE_NAME, ollama_embed, ollama_generate, to_pgvector
import asyncpg
import json


@dataclass
class Meta:
    department: Optional[str] = None
    year: Optional[int] = None


@dataclass
class Tor:
    text: str
    metadata: Meta
    embedding: list[float]


# -----------------------------
# CONFIG
# -----------------------------
ALPHA = 0.6
TOP_K = 5
CANDIDATE_MULT = 4
NOT_FOUND_THRESHOLD = 0.12

# -----------------------------
# Retrieval + Rerank
# -----------------------------


async def retrieve(query: str, filter: Meta):
    q_emb = await ollama_embed(query)

    rows = await search_candidates(q_emb, filter)
    if not rows:
        return []

    return rows


async def search_candidates(query_emb: float, filter: Meta):
    # convert dataclass to dict
    filter_dict = {
        k: v
        for k, v in asdict(filter).items()
        if v is not None
    }
    conn = await asyncpg.connect(**DB_CONFIG)
    rows = await conn.fetch(
        f"""SELECT text, metadata, 1 - (embedding <=> $1) AS score
            FROM {TABLE_NAME}
            WHERE metadata @> $2
            ORDER BY score
            LIMIT $3;""",
        to_pgvector(query_emb),
        json.dumps(filter_dict),
        TOP_K * CANDIDATE_MULT
    )
    await conn.close()
    return rows

# -----------------------------
# Generate answer
# -----------------------------


async def answer_question(question, top_chunks):
    if not top_chunks or top_chunks[0]["score"] < NOT_FOUND_THRESHOLD:
        return "Not found in document."

    context = "\n\n---\n\n".join(
        [f"{c['text']}" for c in top_chunks]
    )

    prompt = f"""
    - ตอบจากข้อมูลที่ให้เท่านั้น
    - ถ้าไม่มีข้อมูล ให้ตอบว่า "ไม่พบข้อมูลในเอกสาร"
    - ห้ามเดา ห้ามสรุปเกินข้อมูล

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """
    return (await ollama_generate(prompt)).strip()


# -----------------------------
# TF-IDF
# -----------------------------
def cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(json.loads(b), dtype=float)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


async def main():
    q = "วงเงินในการจัดหา"
    results = await retrieve(q, Meta(department="มหาวิทยาลัยราชภัฏวไลยอลงกรณ"))
    print(results)

    print("\n--- Retrieved Chunks ---")
    for r in results:
        print(r["score"])
        print(r["text"][:200], "\n")

    ans = await answer_question(q, results)
    print("\nAnswer:", ans)

if __name__ == "__main__":
    asyncio.run(main())
