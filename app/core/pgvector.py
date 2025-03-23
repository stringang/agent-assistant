import uuid
from contextlib import contextmanager

from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from psycopg_pool import ConnectionPool
from pydantic import model_validator, BaseModel
from langchain_community.embeddings import ModelScopeEmbeddings

class PGVectorConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str
    min_connection: int
    max_connection: int
    pg_bigm: bool = False

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values: dict) -> dict:
        if not values["host"]:
            raise ValueError("config PGVECTOR_HOST is required")
        if not values["port"]:
            raise ValueError("config PGVECTOR_PORT is required")
        if not values["user"]:
            raise ValueError("config PGVECTOR_USER is required")
        if not values["password"]:
            raise ValueError("config PGVECTOR_PASSWORD is required")
        if not values["database"]:
            raise ValueError("config PGVECTOR_DATABASE is required")
        if not values["min_connection"]:
            raise ValueError("config PGVECTOR_MIN_CONNECTION is required")
        if not values["max_connection"]:
            raise ValueError("config PGVECTOR_MAX_CONNECTION is required")
        if values["min_connection"] > values["max_connection"]:
            raise ValueError("config PGVECTOR_MIN_CONNECTION should less than PGVECTOR_MAX_CONNECTION")
        return values


# https://github.com/vanna-ai/vanna/blob/main/src/vanna/pgvector/pgvector.py
class PG_VectorStore():
    def __init__(self):
        self.connection_string = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"
        self.n_results = 10
        # self.pool = self._create_connection_pool()
        
        # 创建 embedding 模型
        self.embedding_function = ModelScopeEmbeddings(
            model_id="iic/nlp_gte_sentence-embedding_chinese-large",
            model_revision="v1.1.0"
        )

        self.documentation_collection = PGVector(
            embeddings=self.embedding_function,
            collection_name="my_docs",
            connection=self.connection_string,
        )

    def _create_connection_pool(self):
        pass

    def add_documentation(self, documentation: str, **kwargs) -> str:
        _id = str(uuid.uuid4()) + "-doc"
        doc = Document(
            page_content=documentation,
            metadata={"id": _id},
        )
        self.documentation_collection.add_documents([doc], ids=[doc.metadata["id"]])
        return _id

    def get_related_documentation(self, question: str, **kwargs) -> list:
        documents = self.documentation_collection.similarity_search(query=question, k=self.n_results)
        return [document.page_content for document in documents]
    