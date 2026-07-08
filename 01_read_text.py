from pathlib import Path

file_path = Path("data/rag.txt")

text = file_path.read_text(encoding="utf-8")

print(text)
print("字符数量：", len(text))