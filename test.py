"""
Async RAG: asyncpg + pgvector + Ollama
Suitable for Thai documents / Official PDFs
"""

import json
import asyncio
import asyncpg
import re
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import ollama
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# CONFIG
# -----------------------------
DB_CONFIG = {
    "user": "postgres",
    "password": "P@ssw0rd",
    "database": "rag_db",
    "host": "localhost",
    "port": 5432
}

TABLE_NAME = "doc_chunks"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 80
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.1"
TOP_K = 5
CANDIDATE_MULT = 4
ALPHA = 0.6
NOT_FOUND_THRESHOLD = 0.12
FILE_NAME = "tor2.pdf"


# -----------------------------
# Thai cleanup
# -----------------------------
THAI_MARKS = "่้๊๋ิีึืุูั็์ํ"


def clean_thai_text(text: str) -> str:
    text = text.replace('\u00A0', ' ').replace('\u200b', '')
    text = re.sub(r"([ก-ฮ])\s+([ก-ฮ])", r'\1\2', text)
    text = re.sub(r"([ก-ฮ])\s([' + THAI_MARKS + '])", r'\1\2', text)
    text = re.sub(r"([' + THAI_MARKS + '])\s([ก-ฮ])", r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# Chunking
# -----------------------------
def extract_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []
    for _, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        t = clean_thai_text(t)
        pages.append(t)
    return pages


def chunk_pages(pages):
    chunks = []
    chunk_id = 0
    for text in pages:
        words = text.split()
        i = 0
        while i < len(words):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append({
                    "file_name": FILE_NAME,
                    "chunk_id": f"{FILE_NAME}-{chunk_id}",
                    "text": chunk_text
                })
                chunk_id += 1

            i += (CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks


# -----------------------------
# Async Ollama
# -----------------------------
_executor = ThreadPoolExecutor()


async def ollama_embed(text: str, model=EMBED_MODEL):
    def _run():
        return ollama.embed(model=model, input=text)

    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(_executor, _run)

    vec = resp.get("embeddings") or resp.get("embedding")
    if isinstance(vec, list) and isinstance(vec[0], list):
        return vec[0]
    return vec


async def ollama_generate(prompt: str, model=LLM_MODEL):
    def _run():
        return ollama.generate(model=model, prompt=prompt)

    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(_executor, _run)
    return resp.get("response") or ""


# -----------------------------
# PostgreSQL (asyncpg)
# -----------------------------
async def init_db(dim: int):
    conn = await asyncpg.connect(**DB_CONFIG)
    await conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        file_name TEXT,
        chunk_id TEXT UNIQUE,
        text TEXT,
        embedding VECTOR({dim})
    );
    """)
    await conn.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_emb 
    ON {TABLE_NAME}
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
    """)
    await conn.close()


async def insert_chunks(chunks, embeddings):
    conn = await asyncpg.connect(**DB_CONFIG)
    records = [
        (c["file_name"], c["chunk_id"], c["text"], to_pgvector(emb))
        for c, emb in zip(chunks, embeddings)
    ]
    await conn.executemany(
        f"""INSERT INTO {TABLE_NAME} (file_name, chunk_id, text, embedding)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (chunk_id)
            DO UPDATE SET file_name=EXCLUDED.file_name, text=EXCLUDED.text, embedding=EXCLUDED.embedding;""",
        records
    )
    await conn.close()


async def search_candidates(query_emb):
    conn = await asyncpg.connect(**DB_CONFIG)
    rows = await conn.fetch(
        f"""SELECT file_name, chunk_id, text, embedding
            FROM {TABLE_NAME}
            ORDER BY embedding <-> $1
            LIMIT $2;""",
        to_pgvector(query_emb),
        TOP_K * CANDIDATE_MULT,
    )
    await conn.close()
    return rows


# -----------------------------
# TF-IDF
# -----------------------------
def cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(json.loads(b), dtype=float)
    return float(np.dot(a, b) / ((np.linalg.norm(a)*np.linalg.norm(b))+1e-12))


# -----------------------------
# Retrieval + Rerank
# -----------------------------
async def retrieve(query, tfidf_vec, tfidf_matrix, chunks_texts):
    q_emb = await ollama_embed(query)

    rows = await search_candidates(q_emb)
    if not rows:
        return []

    texts = [r["text"] for r in rows]
    embs = [r["embedding"] for r in rows]

    # lexical
    q_tfidf = tfidf_vec.transform([query])
    cand_tfidf = tfidf_vec.transform(texts)
    lex_sims = (q_tfidf @ cand_tfidf.T).toarray()[0]

    # embedding similarity
    emb_sims = np.array([cosine(q_emb, e) for e in embs])

    # normalize 0..1
    def norm(x):
        x = np.array(x)
        if x.max() - x.min() < 1e-9:
            return np.ones_like(x)
        return (x - x.min()) / (x.max() - x.min())

    emb_n = norm(emb_sims)
    lex_n = norm(lex_sims)
    hybrid = ALPHA * emb_n + (1 - ALPHA) * lex_n

    # pick top-k
    idx_top = np.argsort(hybrid)[::-1][:TOP_K]

    results = []
    for idx in idx_top:
        r = rows[idx]
        results.append({
            "file_name": r["file_name"],
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "score": float(hybrid[idx])
        })

    return results


# -----------------------------
# Generate answer
# -----------------------------
async def answer_question(question, top_chunks):
    if not top_chunks or top_chunks[0]["score"] < NOT_FOUND_THRESHOLD:
        return "Not found in document."

    context = "\n\n---\n\n".join(
        [f"[file_name {c['file_name']}] {c['text']}" for c in top_chunks]
    )

    prompt = f"""
ตอบคำถามโดยใช้เฉพาะข้อมูลจาก CONTEXT ถ้าไม่พบให้ตอบว่า "Not found in document."

CONTEXT:
{context}

QUESTION:
{question}
"""
    return (await ollama_generate(prompt)).strip()


# -----------------------------
# Full build index
# -----------------------------
async def build_index(pdf_path):
    pages = extract_pdf(pdf_path)
    chunks = chunk_pages(pages)

    # determine embedding dim from first chunk
    emb0 = await ollama_embed(chunks[0]["text"])
    dim = len(emb0)

    await init_db(dim)

    # embed chunks
    embeddings = []
    for c in chunks:
        emb = await ollama_embed(c["text"])
        embeddings.append(emb)

    await insert_chunks(chunks, embeddings)

    # build TF-IDF
    texts = [c["text"] for c in chunks]
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    tfidf_mat = normalize(tfidf_vec.fit_transform(texts))

    return chunks, texts, tfidf_vec, tfidf_mat

def to_pgvector(vec):
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


# -----------------------------
# MAIN
# -----------------------------
async def main():

    chunks, texts, tfidf_vec, tfidf_mat = await build_index(FILE_NAME)

    q = "สรุป tor กระทรวงคมนาคมให้หน่อย"
    results = await retrieve(q, tfidf_vec, tfidf_mat, texts)

    print("\n--- Retrieved Chunks ---")
    for r in results:
        print(r["chunk_id"], r["score"])
        print(r["text"][:200], "\n")

    ans = await answer_question(q, results)
    print("\nAnswer:", ans)


if __name__ == "__main__":
    asyncio.run(main())
