from haystack_integrations.document_stores.qdrant import (
    QdrantDocumentStore,
)
from haystack_integrations.components.embedders.fastembed import (
    FastembedTextEmbedder,
)
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
)


# 1. 连接 Qdrant
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


# 3. 创建检索器
retriever = QdrantEmbeddingRetriever(
    document_store=document_store,
    top_k=3,
)


print("数据库文档数量：", document_store.count_documents())
print("输入 exit 退出程序")


# 4. 连续输入问题
while True:
    question = input("\n请输入问题：")

    if question == "exit":
        print("程序结束")
        break

    # 把问题转换成向量
    embed_result = text_embedder.run(
        text=question
    )

    query_embedding = embed_result["embedding"]

    # 从 Qdrant 检索
    result = retriever.run(
        query_embedding=query_embedding
    )

    documents = result["documents"]

    if not documents:
        print("没有检索到相关文档。")
        continue

    # 输出检索结果
    print("\n检索结果：")

    for i, document in enumerate(documents, start=1):
        print("\n第", i, "条")
        print("分数：", document.score)
        print("内容：", document.content)
        print("来源：", document.meta)
        print("-" * 30)

