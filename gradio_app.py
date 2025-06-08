import os
import argparse
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def find_model_root(base_dir: str):
    # Kiểm nếu base_dir có config.json (model HF) hoặc pytorch_model.bin
    if os.path.isfile(os.path.join(base_dir, "config.json")) or \
       os.path.isfile(os.path.join(base_dir, "pytorch_model.bin")):
        return base_dir

    # Duyệt các item bên trong base_dir
    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            # Kiểm nếu thư mục con có config.json hoặc pytorch_model.bin
            if os.path.isfile(os.path.join(full_path, "config.json")) or \
               os.path.isfile(os.path.join(full_path, "pytorch_model.bin")):
                return full_path

    # Nếu không tìm thấy, báo lỗi
    raise FileNotFoundError(
        f"Không tìm thấy thư mục chứa model trong '{base_dir}'. "
        "Hãy chắc chắn đã giải nén đúng cấu trúc hoặc kiểm tra bên trong."
    )

def load_local_model(model_dir: str):
    # Tìm thư mục thực sự chứa model
    root = find_model_root(model_dir)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(root)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(root)
    model.eval()

    return tokenizer, model

def classify_movie_review(review: str, tokenizer, model, threshold: float = 0.7):
    if not isinstance(review, str) or review.strip() == "":
        return "Vui lòng nhập một đoạn review hợp lệ."

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding="longest"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    confidence = probs[pred_idx]

    # Nếu độ tin cậy thấp hơn ngưỡng, trả lời mơ hồ
    if confidence < threshold:
        return (
            f"Không chắc chắn về dự đoán (độ tin cậy: {confidence * 100:.1f}%). "
            "Nội dung có thể không phù hợp hoặc mô hình chưa được huấn luyện đủ."
        )

    label_map = {0: "negative", 1: "positive"}
    label = label_map[pred_idx]
    return f"{label} ({confidence * 100:.1f}%)"

def main():
    parser = argparse.ArgumentParser(description="Gradio app to serve sentiment classification model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs",
        help="Path đến thư mục chứa model (sau khi giải nén). Ví dụ: './outputs'."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port để chạy Gradio server."
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục model: {model_dir}")

    print(f"[INFO] Đang tìm model trong thư mục: {model_dir}")
    tokenizer, model = load_local_model(model_dir)
    print("[INFO] Load model và tokenizer thành công.")

    # Hàm wrapper dùng cho Gradio
    def infer(text):
        return classify_movie_review(text, tokenizer, model)

    # Tạo giao diện Gradio
    iface = gr.Interface(
        fn=infer,
        inputs=gr.Textbox(lines=3, placeholder="Nhập đoạn review phim..."),
        outputs="text",
        title="IMDb Sentiment Classifier (Phim)",
        description="Ứng dụng phân loại sentiment (positive/negative) cho review phim."
    )

    # Chạy server
    iface.launch(server_name="0.0.0.0", server_port=args.port)

if __name__ == "__main__":
    main()
