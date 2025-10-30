

# %%
import os
from pdf2image import convert_from_path

# 设置 PDF 文件名和图片输出文件夹
book_name = ""
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

# 读取 PDF 并转为图片
images = convert_from_path(
    book_name,
    dpi=300,
    poppler_path=r"poppler-24.08.0/Library/bin"  # ✅ 根据你实际路径调整
)

# 保存图片到 images 文件夹中
for i, image in enumerate(images):
    image_path = os.path.join(output_folder, f"page_{i+1}.jpg")
    image.save(image_path, "JPEG")



