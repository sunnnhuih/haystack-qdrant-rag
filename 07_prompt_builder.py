from haystack.components.builders import PromptBuilder

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


# 4. 输入问题
question = input("请输入问题：")


# 5. 把问题转换成向量
embed_result = text_embedder.run(
    text=question
)

query_embedding = embed_result["embedding"]


# 6. 检索相关文档
retrieve_result = retriever.run(
    query_embedding=query_embedding
)

documents = retrieve_result["documents"]


if not documents:
    print("没有检索到相关文档。")

else:
    # 7. 编写提示词模板
    template = """
请只根据下面提供的资料回答问题。
如果资料中没有答案，请回答“根据现有资料无法确定”。

资料：
{% for document in documents %}
{{ document.content }}
{% endfor %}

问题：
{{ question }}

答案：
"""

    # 8. 创建 PromptBuilder
    prompt_builder = PromptBuilder(
        template=template,
        required_variables=["documents", "question"]
    )

    # 9. 把文档和问题填入模板
    prompt_result = prompt_builder.run(
        documents=documents,
        question=question
    )

    # 10. 输出最终提示词
    prompt = prompt_result["prompt"]

    print("\n生成的提示词：")
    print("-" * 40)
    print(prompt)
    print("-" * 40)