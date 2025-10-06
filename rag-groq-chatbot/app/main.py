from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import ChatRequest, ChatResponse, IngestResponse
from app.ingest import extract_text_from_pdf, chunk_text
from app.retriever import SimpleTfidf, VectorStore
from app.llm_client import groq_call, web_search
from utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)
app = FastAPI(title="RAG Groq Chatbot")

vectorizer = SimpleTfidf()
store = VectorStore()
corpus = []

@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)):
    global corpus, store, vectorizer
    texts = []
    for file in files:
        text = extract_text_from_pdf(await file.read())
        texts.extend(chunk_text(text))
    corpus.extend(texts)
    vectorizer.fit(corpus)
    embeddings = vectorizer.transform(corpus)
    store.build(embeddings, corpus)
    return IngestResponse(status="ok", ingested_chunks=len(texts))

@app.post("/chat", response_model=ChatResponse)
async def chat(question: str = Form(...)):
    if not store.embeddings is not None:
        raise HTTPException(status_code=400, detail="No documents ingested")
    qvec = vectorizer.transform([question])[0]
    retrieved = store.retrieve(qvec, top_k=4)
    context = "\n".join([t for t, _ in retrieved])
    prompt = f"Answer based on context:\n{context}\n\nQuestion: {question}"
    answer = groq_call(prompt)
    return ChatResponse(answer=answer, retrieved=retrieved)
