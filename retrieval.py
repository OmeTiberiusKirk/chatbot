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
    "database": "postgres",
    "host": "127.0.0.1",
    "port": 5432,
}

TABLE_NAME = "tors"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5"

FILE_NAME = "tor1.pdf"
METADATA = {
    "title": "ขอบเขตของงาน (Terms of Reference : TOR จ้างบริการบำรุงรักษาและซ่อมแซมแก้ไขระบบงานอิเล็กทรอนิกส์",
    "department": "ปลัดกระทรวงคมนาคม",
    "year": 2024,
    "source": "tor1.pdf"
}

# FILE_NAME = "tor2.pdf"
# METADATA = {
#     "title": "ขอบเขตของงาน (Terms Of Reference : TOR) จ้างพัฒนาระบบคลังข้อสอบและชุดข้อสอบเพื่อประเมินสมิทธิภาพทางภาษาอังกฤษ",
#     "department": "มหาวิทยาลัยราชภัฏวไลยอลงกรณ",
#     "year": 2024,
#     "source": "tor2.pdf"
# }

# -----------------------------
# Thai cleanup
# -----------------------------
THAI_MARKS = "่้๊๋ิีึืุูั็์ํเาะโไแใ์ำ"


def clean_thai_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"([ก-ฮ])\s+([ก-ฮ])", r"\1\2", text)
    text = re.sub(rf"([ก-ฮ])\s([{THAI_MARKS}])", r"\1\2", text)
    text = re.sub(rf"({THAI_MARKS}])\s([ก-ฮ])", r"\1\2", text)
    text = re.sub(rf"({THAI_MARKS}])\s([{THAI_MARKS}])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def thai_sentence_split(text: str):
    # แบ่งโดยใช้ newline และ punctuation ไทย/สากล เป็นหลัก
    # (ไม่ใช้ tokenizer ชั้นสูงเพื่อหลีกเลี่ยง dependency)
    pieces = re.split(r'([\n]+|[।\.\?\!])+', text)
    # รวมชิ้นที่เป็นเนื้อหา
    out = []
    buffer = ""
    for p in pieces:
        if not p:
            continue
        buffer += p
        # ถ้าจบด้วยเครื่องหมายจบประโยคหรือ newline ให้เป็นประโยค
        if re.search(r'[।\.\?\!]\s*$', p) or '\n' in p:
            s = buffer.strip()
            if s:
                out.append(s)
            buffer = ""
    if buffer.strip():
        out.append(buffer.strip())
    return out


# -----------------------------
# Chunking
# -----------------------------
def extract_pdf(pdf_path: str) -> list[tuple[int, str]]:
    # -- coding: utf-8 -- #
    reader = PdfReader(pdf_path)
    pages = []
    for pageno, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        t = clean_thai_text(t)
        pages.append((pageno, t))
    return pages  # list of (page_number, text_of_page)


def chunk_texts(pages: list[tuple[int, str]]):
    chunks = []
    chunk_id = 0
    for pageno, text in pages:
        sentences = thai_sentence_split(text)
        # join sentences into windows of approx chunk_size words (word ~ token)
        words = " ".join(sentences).split()
        i = 0
        while i < len(words):
            chunk_words = words[i: i + CHUNK_SIZE]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append(
                    {
                        "page": pageno,
                        "metadata": METADATA,
                        "text": chunk_text,
                    }
                )
                chunk_id += 1

            i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# -----------------------------
# Async Ollama
# -----------------------------
_executor = ThreadPoolExecutor()


async def ollama_embed(text: str, model=EMBED_MODEL) -> list[float]:
    def _run():
        return ollama.embed(model=model, input=text)

    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(_executor, _run)

    vec = resp.get("embeddings") or resp.get("embedding")
    if isinstance(vec, list) and isinstance(vec[0], list):
        return vec[0]
    return vec


# -----------------------------
# PostgreSQL (asyncpg)
# -----------------------------
async def init_db(dim: int):
    conn = await asyncpg.connect(**DB_CONFIG)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    await conn.execute(
        f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        metadata JSONB,
        embedding VECTOR({dim})
    );
    """
    )
    await conn.execute(
        f"""
    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_emb 
    ON {TABLE_NAME}
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
    """
    )
    await conn.close()


async def insert_chunks(chunks, embeddings):
    conn = await asyncpg.connect(**DB_CONFIG)
    records = [
        (c["text"], json.dumps(c["metadata"]), to_pgvector(emb))
        for c, emb in zip(chunks, embeddings)
    ]

    await conn.executemany(
        f"""INSERT INTO {TABLE_NAME} (text, metadata, embedding)
            VALUES ($1, $2, $3);""",
        records,
    )
    await conn.close()


# -----------------------------
# Full build index
# -----------------------------
async def build_index(pdf_path):
    pages = extract_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages")

    chunks = chunk_texts(pages)
    print(chunks)
    print(f"Created {len(chunks)} chunks")

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


if __name__ == "__main__":
    asyncio.run(main())
