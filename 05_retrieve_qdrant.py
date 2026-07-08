from haystack_integrations.document_stores.qdrant import (
    QdrantDocumentStore,
)
from haystack_integrations.components.embedders.fastembed import (
    FastembedTextEmbedder,
)
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
)


# 1. 连接已经建立好的 Qdrant 文档集合
document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    index="rag_documents",
    embedding_dim=512,
    recreate_index=False,
)


# 2. 创建问题向量化组件
text_embedder = FastembedTextEmbedder(
    model="BAAI/bge-small-zh-v1.5"
)

text_embedder.warm_up()


while True:
    query = input("请输入问题：")

    if query == "exit":
        break

    # 问题向量化
    # Qdrant 检索
    # 打印结果


    # 4. 把问题转换成向量
    embed_result = text_embedder.run(
        text=query
    )

    query_embedding = embed_result["embedding"]


    # 5. 创建检索器
    retriever = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=3,
    )


    # 6. 根据问题向量检索文档
    result = retriever.run(
        query_embedding=query_embedding
    )


    # 7. 输出检索结果
    if not result["documents"]:
        print("没有检索到相关文档。")
    else:
        for document in result["documents"]:
            print("分数：", document.score)
            print("内容：", document.content)
            print("来源：", document.meta)
            print("-" * 30)

