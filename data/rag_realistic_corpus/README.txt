星云知识助手 - 真实场景 RAG 测试语料

目录：
- txt/：5 份 UTF-8 文本文档，可直接上传到你的系统
- pdf/：与 TXT 内容一致的可检索 PDF
- manifest.json：字符数和预计分块数

推荐使用方式：
1. 先清空当前 Qdrant 测试集合。
2. 第一轮只上传 txt/ 中的 5 个文件。
3. 第二轮单独清空集合，再上传 pdf/ 中的 5 个文件。
4. 不要把同内容的 TXT 和 PDF 同时上传，否则会产生重复块。
5. 当前参数 chunk_size=450、chunk_overlap=80 时，预计共生成约 38 个块。

语料特点：
- 同一企业知识助手主题
- 包含跨文档术语和规则
- 包含精确版本号、阈值、角色和异常场景
- 适合测试 Dense、BM25、Hybrid、Top K、过滤、拒答和多文档问答
- 公司、人物和业务数据均为虚构，不包含真实隐私

技术依据：
- Haystack 官方文档
- Qdrant 官方文档
- Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Ragas 官方指标文档

注意：这些文档是原创教学语料，不是从网站整篇复制的材料。
