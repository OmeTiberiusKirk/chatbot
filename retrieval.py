from dataclasses import dataclass
import re
import numpy as np
from pypdf import PdfReader
import chromadb
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# -----------------------------
# 1) Clean-up ข้อความไทย (ปรับปรุงจากเวอร์ชันก่อน)
# -----------------------------
THAI_MARKS = "่้๊๋ิีึืุูั็์ํ"


def clean_thai_text(text: str) -> str:
    # แทนที่ NBSP และ Unicode space ต่าง ๆ
    text = text.replace('\u00A0', ' ').replace('\u200b', '')
    # ลบช่องว่างระหว่างพยัญชนะไทย เช่น "ก า ร" -> "การ"
    text = re.sub(r'([ก-ฮ])\s+([ก-ฮ])', r'\1\2', text)
    # ลบช่องว่างระหว่างพยัญชนะ + สระ/วรรณยุกต์
    text = re.sub(r"([ก-ฮ])\s([' + THAI_MARKS + '])", r'\1\2', text)
    text = re.sub(r"([' + THAI_MARKS + '])\s([ก-ฮ])", r'\1\2', text)
    # ลดช่องว่างต่อเนื่องเป็นช่องเดียว
    text = re.sub(r'\s+', ' ', text)
    # ลบ space ก่อนเครื่องหมายวรรคตอน
    text = re.sub(r'\s+([,.:;!?])', r'\1', text)
    return text.strip()

# -----------------------------
# 2) อ่าน PDF (text-based)
# -----------------------------


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for pageno, page in enumerate(reader.pages, start=1):
        t = page.extract_text() or ""
        t = clean_thai_text(t)
        if t:
            # เก็บหน้าไว้เป็น metadata ด้วย
            pages.append((pageno, t))
    return pages  # list of (page_number, text_of_page)

# -----------------------------
# 3) Sentence split + chunking with overlap
# -----------------------------


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


def chunk_texts(pages, chunk_size=300, overlap=50):
    """
    pages: list of (pageno, text)
    returns: list of dicts: {id, page, chunk_text}
    """
    chunks = []
    chunk_id = 0
    for page_no, text in pages:
        sentences = thai_sentence_split(text)
        # join sentences into windows of approx chunk_size words (word ~ token)
        words = " ".join(sentences).split()
        i = 0
        while i < len(words):
            chunk_words = words[i: i + chunk_size]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunks.append({
                    "id": f"{page_no}-{chunk_id}",
                    "page": page_no,
                    "text": chunk_text
                })
                chunk_id += 1
            i += (chunk_size - overlap)
    return chunks


# -----------------------------
# 4) Embedding via Ollama (batch-safe)
# -----------------------------
ollama_client = ollama.Client()


def embed_texts(texts, model="mxbai-embed-large"):
    """
    texts: list of strings
    returns: numpy array shape (n, d)
    """
    embeddings = []
    # Ollama API: client.embed(model=..., input=...)
    for t in texts:
        resp = ollama_client.embed(model=model, input=t)
        vec = resp.get("embeddings") or resp.get("embedding")  # be tolerant
        if not vec:
            raise RuntimeError(
                "Embedding failed or returned empty for a chunk.")
        embeddings.append(vec[0] if isinstance(vec, list)
                          and isinstance(vec[0], list) else vec)
    return np.array(embeddings, dtype=np.float32)

# -----------------------------
# 5) Build Chroma collection (persistent)
# -----------------------------


def init_chroma(persist_dir="./chroma_db", collection_name="thai_tor"):
    settings = chromadb.config.Settings(
        # chroma_db_impl="duckdb+parquet",
        persist_directory=persist_dir
    )
    client = chromadb.Client(settings)
    coll = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"})
    return client, coll

# -----------------------------
# 6) Build TF-IDF vectorizer (lexical retrieval)
# -----------------------------


def build_tfidf_index(texts):
    # texts: list of documents (chunk text)
    vec = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2), max_features=20000)
    X = vec.fit_transform(texts)
    return vec, normalize(X, axis=1)

# -----------------------------
# 7) Hybrid retrieval: combine embedding cosine + TF-IDF cosine
# -----------------------------


def cosine_sim(a, b):
    # a: 1d, b: nxd
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)


def retrieve_hybrid(
    query,
    tfidf_vec,
    tfidf_matrix,
    chunks_meta,
    chroma_collection: chromadb.Collection,
    top_k=5,
    emb_model="mxbai-embed-large",
    alpha=0.6
):
    """
    alpha: weight for embedding score (0..1). lexical weight = (1-alpha)
    """
    # 1) embed query
    q_emb = embed_texts([query], model=emb_model)[0]  # 1D

    # 2) embedding search via chroma (fast approximate) to get candidate ids
    chroma_res = chroma_collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k*3,
        include=['distances', 'metadatas', 'documents']
    )
    # chroma returns distances for cosine (lower better) depending on impl — we'll compute cosine separately below
    candidate_ids = chroma_res["ids"][0] if chroma_res["ids"] else []
    # map candidate id -> chunk index
    id_to_idx = {chunks_meta[i]["id"]: i for i in range(len(chunks_meta))}
    candidate_idx = [id_to_idx[cid]
                     for cid in candidate_ids if cid in id_to_idx]

    # 3) compute embedding cosine similarities for candidates
    all_embeddings = np.array([m.get("embedding") for m in chroma_collection.get(
        include=['metadatas'])]) if False else None
    # safer: we already computed embeddings before and saved in chroma; but here recompute using stored collection documents
    # For simplicity compute embedding similarity by embedding the chunk texts here (costly) — better to persist embeddings separately in your pipeline.
    # We'll compute embeddings for candidate chunks by asking the collection for documents and embedding them locally, OR
    # use the distances returned (if represent cosine). To be robust, compute using our saved chunks_meta embeddings if we stored them.

    # --- assume we have stored embeddings in chunks_meta as 'embedding' when building index ---
    chunk_embs = np.array([c.get("embedding") for c in chunks_meta])
    emb_sims = cosine_sim(q_emb, chunk_embs)  # sims to all
    # 4) lexical TF-IDF score
    q_tfidf = tfidf_vec.transform([query])
    lex_sims = (q_tfidf.dot(tfidf_matrix.T).toarray())[0] # similarity to all chunks

    # 5) combine scores and select top_k
    # normalize both scores to 0..1
    def minmax(x): return (x - x.min()) / (x.max() - x.min() + 1e-12)
    emb_norm = minmax(emb_sims)
    lex_norm = minmax(lex_sims)
    hybrid = alpha * emb_norm + (1 - alpha) * lex_norm

    top_idx = np.argsort(hybrid)[::-1][:top_k]
    results = []
    for idx in top_idx:
        results.append({
            "id": chunks_meta[idx]["id"],
            "page": chunks_meta[idx]["page"],
            "text": chunks_meta[idx]["text"],
            "score": float(hybrid[idx]),
            "emb_score": float(emb_norm[idx]),
            "lex_score": float(lex_norm[idx])
        })
    return results

# -----------------------------
# 8) Answer generation with Ollama LLM (prompting + "Not found" logic)
# -----------------------------


def generate_answer_with_context(
    question,
    top_chunks,
    model="llama3.1",
    not_found_threshold=0.12
):
    """
    top_chunks: list of dicts with keys text, page, score
    not_found_threshold: if highest hybrid score < threshold => reply "Not found in document."
    """
    if not top_chunks:
        return "Not found in document."

    top_score = top_chunks[0]["score"]
    if top_score < not_found_threshold:
        return "Not found in document."

    # assemble context (limit length)
    context_parts = []
    for c in top_chunks:
        context_parts.append(f"[page {c['page']}] {c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
Answer the question using ONLY the context below.
If the answer cannot be found in the context, reply exactly: "Not found in document."

CONTEXT:
{context}

QUESTION:
{question}

Provide a concise, factual answer in Thai (or say "Not found in document.").
"""
    resp = ollama_client.generate(model=model, prompt=prompt)
    # Ollama returns {"response": "..."} or similar
    answer = resp.get("response") or resp.get(
        "text") or resp.get("content") or ""
    return answer.strip()

# -----------------------------
# 9) Full pipeline helper to build index from PDF
# -----------------------------


def build_index_from_pdf(
    pdf_path,
    persist_dir="./chroma_db",
    collection_name="thai_tor",
    chunk_size=350,
    overlap=70,
    emb_model="mxbai-embed-large"
):
    @dataclass
    class RETURN:
        client: any
        collection: chromadb.Collection
        chunks_meta: any
        tfidf_vec: any
        tfidf_matrix: any

    # extract
    pages = extract_pdf_text(pdf_path)
    print(f"Loaded {len(pages)} pages")

    # chunk
    chunks = chunk_texts(pages, chunk_size=chunk_size, overlap=overlap)
    print(f"Created {len(chunks)} chunks")

    # embed chunks (may take time)
    texts = [c["text"] for c in chunks]
    print("Embedding chunks via Ollama...")
    embeddings = embed_texts(texts, model=emb_model)
    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()

    # init chroma and add
    client, coll = init_chroma(
        persist_dir=persist_dir, collection_name=collection_name)
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]
    coll.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )
    # client.persist()
    print("Chroma persisted.")

    # build tfidf
    tfidf_vec, tfidf_matrix = build_tfidf_index(documents)
    print("TF-IDF index built.")

    return RETURN(
        client=client,
        collection=coll,
        chunks_meta=chunks,
        tfidf_vec=tfidf_vec,
        tfidf_matrix=tfidf_matrix
    )


# -----------------------------
# 10) Example usage
# -----------------------------
if __name__ == "__main__":
    # ตัวอย่างรัน (ปรับ path ให้ตรง)
    pdf_path = "tor.pdf"  # เปลี่ยนตามตำแหน่งไฟล์
    state = build_index_from_pdf(pdf_path)

    # คำถามตัวอย่าง
    q = "ระยะเวลาการให้บริการของสัญญาคือเมื่อไหร่"
    results = retrieve_hybrid(
        q,
        state.tfidf_vec,
        state.tfidf_matrix,
        state.chunks_meta,
        state.collection,
        top_k=5,
        alpha=0.6
    )
    print("Top chunks (hybrid):")
    for r in results:
        print(r["id"], r["page"], r["score"], r["text"][:200])
    answer = generate_answer_with_context(q, results)
    print("Answer:")
    print(answer)
