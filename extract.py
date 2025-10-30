import os
from ultralytics import YOLO  # 确保已安装 ultralytics
from PIL import Image
import cv2
import json
import re




def natural_key(s):
    """
    实现自然排序，比如 page_2.jpg < page_10.jpg
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]



from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
    use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
)
# ocr = PaddleOCR(lang="en") # 通过 lang 参数来使用英文模型
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # 通过 ocr_version 参数来使用 PP-OCR 其他版本
# ocr = PaddleOCR(device="gpu") # 通过 device 参数使得在模型推理时使用 GPU
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_server_det",
#     text_recognition_model_name="PP-OCRv5_server_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# ) # 更换 PP-OCRv5_server 模型



# === 配置 ===
model_path = "runs/detect/train2/weights/best.pt"
image_folder = "images"  # 存放原图的文件夹
output_jsonl = "ocr_results.jsonl"
output_crop_folder = "cropped_images"  # 保存裁剪图像的路径
os.makedirs(output_crop_folder, exist_ok=True)
output_blur_folder = "blurred_images"
os.makedirs(output_blur_folder, exist_ok=True)

# 加载模型
model = YOLO(model_path)

# 遍历所有图片
file_list = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
file_list.sort(key=natural_key)

with open(output_jsonl, "w", encoding="utf-8") as fout:
    for filename in file_list:
        image_path = os.path.join(image_folder, filename)
        img_pil = Image.open(image_path)
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print(f"❌ 无法读取图片: {filename}")
            continue

        # YOLO目标检测
        results = model.predict(source=image_path, conf=0.6, save=False)
        boxes = results[0].boxes.xyxy

        # === 模糊图像中每个目标区域 ===
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])

            # 裁剪保存
            cropped = img_pil.crop((x1, y1, x2, y2))
            crop_path = os.path.join(output_crop_folder, f"{os.path.splitext(filename)[0]}_crop{i}.jpg")
            cropped.save(crop_path)

            # 模糊处理
            roi = img_cv[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (85, 85), 0)
            img_cv[y1:y2, x1:x2] = blurred_roi

        # === 裁掉顶部10%后保存整图供OCR使用 === 去掉页码页头（页码影响ocr精度）
        h = img_cv.shape[0]
        cropped_blurred_img = img_cv[int(h * 0.10):, :]  # 裁掉顶部10%
        blurred_path = os.path.join(output_blur_folder, filename)
        cv2.imwrite(blurred_path, cropped_blurred_img)

        # === OCR 模糊图像并写入 JSONL ===
        result = ocr.predict(blurred_path)  # OCR 模糊后图像

        for res in result:
            texts = res.rec_texts if hasattr(res, "rec_texts") else res.get("rec_texts", [])
            full_text = "".join(texts)
            cleaned_text = re.sub(r"photo\s*0\.\d{1,3}", "", full_text, flags=re.IGNORECASE).strip()

            json_line = {
                "image": filename,
                "cleaned_text": cleaned_text
            }

            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")





#去掉空集
input_path = "ocr_results.jsonl"
output_path = "ocr_results_filtered.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            item = json.loads(line.strip())
            text = item.get("cleaned_text", "").strip()
            if len(text) >= 15:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        except json.JSONDecodeError:
            print("跳过无法解析的行：", line)

