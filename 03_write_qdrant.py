from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.writers import DocumentWriter

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.fastembed import (
    FastembedDocumentEmbedder,
)

document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    index="rag_documents",
    embedding_dim=512,
    recreate_index=True,
)

documents = [
    Document(
        content="本项目使用 Qdrant 作为向量数据库。",
        meta={
            "source": "rag.txt",
            "chunk_id": 0,
        },
    )
]

embedder = FastembedDocumentEmbedder(
    model="BAAI/bge-small-zh-v1.5"
)

embedder.warm_up()

result = embedder.run(documents=documents)

writer = DocumentWriter(
    document_store=document_store,
    policy=DuplicatePolicy.OVERWRITE,
)

writer.run(documents=result["documents"])

print("数据库文档数量：", document_store.count_documents())