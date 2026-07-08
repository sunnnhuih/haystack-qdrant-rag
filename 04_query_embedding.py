from haystack_integrations.components.embedders.fastembed import (
    FastembedTextEmbedder,
)

text_embedder = FastembedTextEmbedder(
    model="BAAI/bge-small-zh-v1.5"
)

text_embedder.warm_up()

result = text_embedder.run(
    text="本项目使用什么数据库？"
)

query_embedding = result["embedding"]

print("问题向量维度：", len(query_embedding))