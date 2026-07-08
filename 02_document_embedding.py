from haystack import Document
from haystack_integrations.components.embedders.fastembed import (
    FastembedDocumentEmbedder,
)

documents = [
    Document(content="RAG 的中文名称是检索增强生成。"),
    Document(content="本项目使用 Qdrant 作为向量数据库。"),
]

embedder = FastembedDocumentEmbedder(
    model="BAAI/bge-small-zh-v1.5"
)

embedder.warm_up()

result = embedder.run(documents=documents)

embedded_documents = result["documents"]

for document in embedded_documents:
    print(document.content)
    print("向量维度：", len(document.embedding))
    print(document.embedding[:5])