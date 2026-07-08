from haystack_integrations.document_stores.qdrant import (
    QdrantDocumentStore,
)

document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    index="rag_documents",
    embedding_dim=512,
    recreate_index=True,
)

print("数据库已清空")
print("当前文档数量：", document_store.count_documents())