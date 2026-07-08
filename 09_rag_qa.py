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
from haystack_integrations.components.generators.ollama import (
    OllamaGenerator,
)


# 1. 连接已经建立好的 Qdrant 集合
document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    index="rag_documents",
    embedding_dim=512,
    recreate_index=False,
)

document_count = document_store.count_documents()
print("数据库文档数量：", document_count)

if document_count == 0:
    print("数据库为空，请先运行 03_write_qdrant.py")
    raise SystemExit


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


# 4. 创建提示词模板
template = """
请只根据下面提供的资料回答问题。
不要使用资料以外的知识。
如果资料中没有答案，请回答“根据现有资料无法确定”。

资料：
{% for document in documents %}
{{ document.content }}
{% endfor %}

问题：
{{ question }}

答案：
"""

prompt_builder = PromptBuilder(
    template=template,
    required_variables=["documents", "question"]
)


# 5. 创建本地大模型组件
generator = OllamaGenerator(
    model="qwen2.5:3b",
    url="http://localhost:11434",
    generation_kwargs={
        "temperature": 0.1,
        "num_predict": 200,
    },
)


# 6. 输入问题
question = input("请输入问题：")


# 7. 把问题转换成向量
embed_result = text_embedder.run(
    text=question
)

query_embedding = embed_result["embedding"]


# 8. 从 Qdrant 检索相关文档
retrieve_result = retriever.run(
    query_embedding=query_embedding
)

documents = retrieve_result["documents"]

if not documents:
    print("没有检索到相关文档。")
    raise SystemExit


# 9. 把资料和问题拼成提示词
prompt_result = prompt_builder.run(
    documents=documents,
    question=question
)

prompt = prompt_result["prompt"]


# 10. 把提示词交给大模型
generate_result = generator.run(
    prompt=prompt
)

answer = generate_result["replies"][0]


# 11. 输出答案
print("\n回答：")
print(answer)

print("\n检索来源：")

for i, document in enumerate(documents, start=1):
    print("第", i, "条：", document.meta)
    print("相似度：", document.score)