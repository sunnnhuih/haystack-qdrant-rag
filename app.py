
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, UploadFile, File
from haystack.dataclasses import ByteStream
from haystack.components.converters import PyPDFToDocument
from haystack import Pipeline, Document
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import (
    FastembedDocumentEmbedder,
)
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
from fastapi.responses import FileResponse

# ==============================
# 1. 创建 FastAPI 应用
# ==============================

app = FastAPI(
    title="Haystack RAG API",
    description="基于 Haystack、Qdrant 和 Ollama 的问答接口",
    version="1.0"
)


# ==============================
# 2. 定义请求数据格式
# ==============================

class AskRequest(BaseModel):
    question: str


# ==============================
# 3. 连接 Qdrant
# ==============================

document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    index="rag_documents",
    embedding_dim=512,
    recreate_index=False,
)


# ==============================
# 文档分块组件
# ==============================

document_splitter = RecursiveDocumentSplitter(
    split_length=450,
    split_overlap=80,
    split_unit="char",
    separators=[
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
    ],
)
pdf_converter = PyPDFToDocument()

# ==============================
# 文档向量化组件
# ==============================

document_embedder = FastembedDocumentEmbedder(
    model="BAAI/bge-small-zh-v1.5"
)

document_embedder.warm_up()


# ==============================
# 文档写入组件
# ==============================

document_writer = DocumentWriter(
    document_store=document_store,
    policy=DuplicatePolicy.OVERWRITE,
)


# ==============================
# 4. 创建 Haystack 组件
# ==============================

text_embedder = FastembedTextEmbedder(
    model="BAAI/bge-small-zh-v1.5"
)

text_embedder.warm_up()


retriever = QdrantEmbeddingRetriever(
    document_store=document_store,
    top_k=3,
    score_threshold=0.45,
)


template = """
请严格根据下面的资料回答问题。

要求：
1. 只使用与问题直接对应的资料。
2. 如果资料中存在相同或相近的问题及“答：”，优先依据该答案作答。
3. 不要综合其他题目的内容。
4. 不要补充资料之外的常识。
5. 资料没有明确答案时，回答“根据现有资料无法确定”。

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


# ==============================
# 5. 创建 Pipeline
# ==============================

rag_pipeline = Pipeline()

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


# ==============================
# 6. 连接 Pipeline
# ==============================

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


# ==============================
# 7. 健康检查接口
# ==============================

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents": document_store.count_documents()
    }

# ==============================
# 上传
# ==============================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    lower_filename = filename.lower()

    if not (
        lower_filename.endswith(".txt")
        or lower_filename.endswith(".pdf")
    ):
        raise HTTPException(
            status_code=400,
            detail="目前只支持 TXT 和 PDF 文件"
        )

    try:
        file_bytes = await file.read()

        # ==========================
        # 1. TXT 转 Document
        # ==========================
        if lower_filename.endswith(".txt"):
            try:
                text = file_bytes.decode("utf-8-sig").strip()
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="TXT 文件不是 UTF-8 编码"
                )

            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="TXT 文件内容为空"
                )

            original_documents = [
                Document(
                    content=text,
                    meta={
                        "source": filename,
                        "file_name": filename,
                        "file_type": "txt"
                    }
                )
            ]

        # ==========================
        # 2. PDF 转 Document
        # ==========================
        else:
            pdf_stream = ByteStream(
                data=file_bytes,
                mime_type="application/pdf",
                meta={
                    "source": filename,
                    "file_name": filename,
                    "file_type": "pdf"
                }
            )

            convert_result = pdf_converter.run(
                sources=[pdf_stream]
            )

            original_documents = convert_result["documents"]

            # 删除没有提取到文字的 Document
            original_documents = [
                document
                for document in original_documents
                if document.content
                and document.content.strip()
            ]

            if not original_documents:
                raise HTTPException(
                    status_code=400,
                    detail="PDF 没有提取到文字，可能是扫描版 PDF"
                )

        # ==========================
        # 3. 文档分块
        # ==========================
        split_result = document_splitter.run(
            documents=original_documents
        )

        split_documents = split_result["documents"]

        if not split_documents:
            raise HTTPException(
                status_code=400,
                detail="文件没有生成有效分块"
            )

        # 给每个分块补充分块编号
        for index, document in enumerate(split_documents):
            document.meta["chunk_id"] = index
            document.meta["source"] = filename
            document.meta["file_name"] = filename

        # ==========================
        # 4. 生成文档向量
        # ==========================
        embed_result = document_embedder.run(
            documents=split_documents
        )

        embedded_documents = embed_result["documents"]

        # ==========================
        # 5. 写入 Qdrant
        # ==========================
        write_result = document_writer.run(
            documents=embedded_documents
        )

        return {
            "message": "文件写入知识库成功",
            "filename": filename,
            "file_type": (
                "pdf"
                if lower_filename.endswith(".pdf")
                else "txt"
            ),
            "chunks": len(split_documents),
            "documents_written": write_result["documents_written"],
            "total_documents": document_store.count_documents()
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==============================
# 8. RAG 问答接口
# ==============================

@app.post("/ask")
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="问题不能为空"
        )

    if document_store.count_documents() == 0:
        raise HTTPException(
            status_code=503,
            detail="知识库为空，请先运行 03_write_qdrant.py"
        )

    try:
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

        answer = result["generator"]["replies"][0]
        documents = result["retriever"]["documents"]

        sources = []

        for document in documents:
            sources.append(
                {
                    "content": document.content,
                    "score": document.score,
                    "meta": document.meta
                }
            )

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )