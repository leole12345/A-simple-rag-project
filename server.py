from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import faiss
import json
import torch
import uvicorn
import os

# 初始化 FastAPI
app = FastAPI()

# 加载模型
from transformers import AutoTokenizer, AutoModel
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("./bge-large-zh-v1.5")
model = AutoModel.from_pretrained("./bge-large-zh-v1.5").to("cuda")

# 加载 FAISS 索引和元数据
index = faiss.read_index("faiss_bge_big.index")
with open("faiss_bge_metadata_big.json", "r", encoding="utf-8") as f:
    metadatas = json.load(f)

# 定义请求数据结构
class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5

# 获取 embedding
@torch.no_grad()
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

IMAGE_DIR = "cropped_images"
def list_all_crops_for_page(page):
    """返回该页所有 crop 图像的 url 列表"""
    pattern = f"page_{page}_crop"
    files = os.listdir(IMAGE_DIR)
    urls = [
        f"http://127.0.0.1:8000/images/{f}"
        for f in files if f.startswith(pattern)
    ]
    return sorted(urls)



# 检索接口
@app.post("/retrieval")
async def retrieve(req: RetrievalRequest):
    q_vec = get_embedding(req.query).reshape(1, -1)
    D, I = index.search(q_vec.astype("float32"), req.top_k)

    chunks = []
    for i in I[0]:
        # 获取中心句
        center = i
        window_size = 2  # 可调整窗口大小
        left = max(0, center - window_size)
        right = min(len(metadatas), center + window_size + 1)

        # 拼接窗口内句子
        window_texts = [metadatas[j]["text"] for j in range(left, right)]
        combined = "。".join(window_texts) + "。"

        # 返回中心句元信息 + 扩展上下文
        center_meta = metadatas[center]
        page = center_meta["page"]
        chunks.append({
            "page": page,
            "sentence_id": center_meta["sentence_id"],
            "text": center_meta["text"],
            "context": combined,
            "image_urls": list_all_crops_for_page(page)
        })

    return JSONResponse(content={"chunks": chunks})


# 图片服务接口
IMAGE_DIR = "cropped_images"

@app.get("/images/{filename}")
async def serve_image(filename: str):
    file_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(file_path)


# 运行
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
