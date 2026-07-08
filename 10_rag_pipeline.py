from haystack import Pipeline
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


# 1. 连接已经存在的 Qdrant 集合
document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    index="rag_documents",
    embedding_dim=512,
    recreate_index=False,
)

print("数据库文档数量：", document_store.count_documents())

if document_store.count_documents() == 0:
    print("数据库为空，请先运行 03_write_qdrant.py")
    raise SystemExit


# 2. 创建各个组件
text_embedder = FastembedTextEmbedder(
    model="BAAI/bge-small-zh-v1.5"
)

text_embedder.warm_up()

retriever = QdrantEmbeddingRetriever(
    document_store=document_store,
    top_k=3,
)

template = """
请只根据下面提供的资料回答问题。
不要使用资料以外的内容。
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

generator = OllamaGenerator(
    model="qwen2.5:3b",
    url="http://localhost:11434",
    generation_kwargs={
        "temperature": 0.1,
        "num_predict": 200,
    },
)


# 3. 创建 Pipeline
rag_pipeline = Pipeline()


# 4. 把组件加入 Pipeline
rag_pipeline.add_component(
    "text_embedder",
    text_embedder
)

rag_pipeline.add_component(
    "retriever",
    retriever
)

rag_pipeline.add_component(
    "prompt_builder",
    prompt_builder
)

rag_pipeline.add_component(
    "generator",
    generator
)


# 5. 连接各个组件
rag_pipeline.connect(
    "text_embedder.embedding",
    "retriever.query_embedding"
)

rag_pipeline.connect(
    "retriever.documents",
    "prompt_builder.documents"
)

rag_pipeline.connect(
    "prompt_builder.prompt",
    "generator.prompt"
)


# 6. 输入问题
question = input("请输入问题：")


# 7. 一次运行整个 Pipeline
result = rag_pipeline.run(
    {
        "text_embedder": {
            "text": question
        },
        "prompt_builder": {
            "question": question
        }
    },
    include_outputs_from={"retriever"}
)


# 8. 输出答案
answer = result["generator"]["replies"][0]

print("\n回答：")
print(answer)


# 9. 输出检索来源
documents = result["retriever"]["documents"]

print("\n检索来源：")

for i, document in enumerate(documents, start=1):
    print("第", i, "条")
    print("内容：", document.content)
    print("来源：", document.meta)
    print("相似度：", document.score)
    print("-" * 30)