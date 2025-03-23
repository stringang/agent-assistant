import os

from fastapi import FastAPI
import logging
import uvicorn
logging.basicConfig(level=logging.DEBUG)

from routes import chat
from core import pgvector

vector_store = pgvector.PG_VectorStore()

app = FastAPI()

app.include_router(chat, prefix="")

if __name__ == "__main__":
    # log_conf_path = os.environ.get("LOG_CONFIG")
    uvicorn.run("main:app", host="0.0.0.0", port=8080)


