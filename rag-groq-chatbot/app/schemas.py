from pydantic import BaseModel
from typing import List, Optional

class IngestResponse(BaseModel):
    status: str
    ingested_chunks: int

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    retrieved: Optional[list] = None
