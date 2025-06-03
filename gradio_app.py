import argparse
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def init_model(model_dir="./outputs"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def sentiment_fn(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    idx = np.argmax(probs)
    label_map = {0: "negative", 1: "positive"}
    return f"{label_map[idx]} ({probs[idx]*100:.1f}%)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    tokenizer, model = init_model("./outputs")  # model dir đã được CI/CD giải nén

    def wrapper(text):
        return sentiment_fn(text, tokenizer, model)

    demo = gr.Interface(
        fn=wrapper,
        inputs=gr.Textbox(lines=2, placeholder="Nhập review phim..."),
        outputs="text",
        title="IMDb Sentiment Classifier",
        description="Nhập một câu review, trả về positive/negative kèm xác suất."
    )
    demo.launch(server_name="0.0.0.0", server_port=args.port)

if __name__ == "__main__":
    main()
