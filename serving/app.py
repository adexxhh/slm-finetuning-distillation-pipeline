"""SQL inference API. Unconfigured models fail closed; buffered SSE is labelled."""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from data_engine.validator import SQLValidator
from data_engine.schemas import get_schema_context_prompt
from serving.backend import load_backend

backend=None
slots=asyncio.Semaphore(1)

@asynccontextmanager
async def lifespan(app):
    global backend
    backend=load_backend()
    yield
    backend=None

app=FastAPI(title="Text-to-SQL research prototype",lifespan=lifespan)
validator=SQLValidator()

class SQLRequest(BaseModel):
    query: str=Field(min_length=1,max_length=10000)
    max_tokens: int=Field(default=256,ge=1,le=2048)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str="sql-slm"
    messages:list[Message]=Field(min_length=1)
    stream:bool=False
    max_tokens:int=Field(default=256,ge=1,le=2048)

@app.get("/health")
async def health():
    return {"status":"ready" if backend else "unconfigured","model_loaded":backend is not None,
            "backend":backend.name if backend else None,"streaming":"buffered, not live token streaming"}

async def predict(question,max_tokens):
    if backend is None: raise HTTPException(503,"Set MODEL_PATH to a downloaded local checkpoint; ADAPTER_PATH is optional")
    async with slots:
        result=await asyncio.to_thread(backend.generate,question,get_schema_context_prompt(),max_tokens)
    valid,error=validator.validate(result["sql"])
    if not valid: raise HTTPException(422,{"error":"Generated SQL rejected","reason":error})
    return result

@app.post("/predict/sql")
async def predict_sql(req:SQLRequest):
    return await predict(req.query,req.max_tokens)

@app.post("/v1/chat/completions")
async def chat(req:ChatRequest):
    # This SQL service supports a single question, not arbitrary multi-turn chat.
    users=[m.content for m in req.messages if m.role=="user"]
    if len(users)!=1 or any(m.role!="user" for m in req.messages):
        raise HTTPException(422,"Provide exactly one user message; schema is controlled by the server")
    result=await predict(users[0],req.max_tokens)
    if req.stream:
        async def buffered():
            yield "data: "+json.dumps({"choices":[{"delta":{"content":result["sql"]},
                  "finish_reason":"stop"}],"streaming_mode":"buffered"})+"\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(buffered(),media_type="text/event-stream")
    return {"choices":[{"message":{"role":"assistant","content":result["sql"]},"finish_reason":"stop"}],
            "usage":{"completion_tokens":result["total_tokens"]},"backend":result["backend"]}
