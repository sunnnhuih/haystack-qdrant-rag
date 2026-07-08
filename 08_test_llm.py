from haystack_integrations.components.generators.ollama import (
    OllamaGenerator,
)


# 创建本地大模型组件
generator = OllamaGenerator(
    model="qwen2.5:3b",
    url="http://localhost:11434",
    generation_kwargs={
        "temperature": 0.1,
        "num_predict": 200,
    },
)


# 给大模型发送提示词
result = generator.run(
    prompt="请用一句话解释什么是 RAG。"
)


# 输出回答
print(result["replies"][0])