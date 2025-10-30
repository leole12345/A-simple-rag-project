# %%
import json
import re
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("./bge-large-zh-v1.5")
model = AutoModel.from_pretrained("./bge-large-zh-v1.5").to("cuda")
model.eval()

# %%


# %%
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


# %%
# === Step 1: 加载 OCR 文本 ===
input_path = "ocr_results_filtered.jsonl"
with open(input_path, "r", encoding="utf-8") as f:
    pages = [json.loads(line.strip()) for line in f if line.strip()]

# === Step 2: 切分为句子级 chunk ===
sentence_chunks = []
for entry in pages:
    image = entry["image"]
    page = int(re.search(r"page_(\d+)", image).group(1))
    text = entry["cleaned_text"]
    sentences = re.split(r"[。！？]", text)  # 可按需要扩展为更丰富标点分句
    for idx, sent in enumerate(sentences):
        sent = sent.strip()
        if len(sent) < 5:
            continue
        sentence_chunks.append({
            "text": sent,
            "page": page,
            "sentence_id": idx
        })

# %%

# === Step 3: 生成向量 ===
@torch.no_grad()
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

vectors = [get_embedding(c["text"]) for c in sentence_chunks]
vectors_np = np.vstack(vectors).astype("float32")

# === Step 4: 构建 FAISS 检索索引 ===
index = faiss.IndexFlatL2(vectors_np.shape[1])
index.add(vectors_np)

# %%

# === Step 5: 模拟检索：返回句窗上下文 ===
def retrieve_with_context(query, top_k=5, window_size=2):
    q_vec = get_embedding(query).reshape(1, -1)
    D, I = index.search(q_vec, top_k)

    results = []
    for i in I[0]:
        center = i
        left = max(0, center - window_size)
        right = min(len(sentence_chunks), center + window_size + 1)
        window = [sentence_chunks[j]["text"] for j in range(left, right)]
        combined = "。".join(window) + "。"
        results.append({
            "context": combined,
            "page": sentence_chunks[center]["page"]
        })
    return results

# %%

# # === Step 6: 测试 ===
# query = "作者是谁"
# res = retrieve_with_context(query)
# for r in res:
#     print(f"\n📄 页码: {r['page']}\n🔍 匹配上下文:\n{r['context']}")


# %%
# 保存 FAISS 向量索引
faiss.write_index(index, "faiss_bge_big.index")

# 保存元数据（chunk 内容 + 页码等）
with open("faiss_bge_metadata_big.json", "w", encoding="utf-8") as f:
    json.dump(sentence_chunks, f, ensure_ascii=False, indent=2)

print("✅ 已保存 FAISS 索引和元数据！")

# %%
with open("faiss_bge_metadata_big.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# 加载 faiss 索引
index = faiss.read_index("faiss_bge_big.index")

# 向量获取函数
@torch.no_grad()
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.base_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

# 时间事件提取函数
def retrieve_events_by_year(year, top_k=20):
    query = f"在{year}年发生了什么？"
    q_vec = get_embedding(query).reshape(1, -1)
    D, I = index.search(q_vec, top_k)

    results = []
    for i in I[0]:
        meta = chunks[i]
        text = meta["text"]
        if re.search(rf"{year}", text):
          results.append(text)

    return "。".join(results)


# 遍历时间范围，提取信息
year_summary = {}
for year in range(1930, 2008):
    context = retrieve_events_by_year(year)
    if context.strip():
        year_summary[str(year)] = context

# 保存输出
with open("yearly_summary.json", "w", encoding="utf-8") as f:
    json.dump(year_summary, f, ensure_ascii=False, indent=2)




