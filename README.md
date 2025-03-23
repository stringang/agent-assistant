# RAG assistant

```shell
# 初始化
poetry init
# 添加依赖
poetry add pendulum
# 安装依赖
poetry install
# 使用 virtual env
poetry run which python
poetry shell

# 运行 pgvector
docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 5432:5432 -d pgvector/pgvector:pg16
```


## Reference

- https://github.com/langchain-ai/rag-from-scratch
- https://datawhalechina.github.io/llm-universe/