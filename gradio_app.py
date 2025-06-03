import os
import argparse
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_local_model(model_path: str):
    """
    Load a fine-tuned Hugging Face Transformer model and its tokenizer
    from a local directory.

    Args:
        model_path (str): Path to the folder containing the model files
                          (e.g., config.json, pytorch_model.bin, tokenizer files).

    Returns:
        tokenizer: AutoTokenizer instance loaded from model_path.
        model: AutoModelForSequenceClassification instance loaded from model_path.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    return tokenizer, model

def classify_movie_review(review: str, tokenizer, model):
    """
    Perform sentiment classification on a single movie review.

    Args:
        review (str): The input text (movie review) to classify.
        tokenizer: The tokenizer loaded from the model directory.
        model: The fine-tuned classification model.

    Returns:
        str: A string label ("positive" / "negative") with probability.
    """
    # Tokenize and prepare inputs
    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding="longest"
    )

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Compute probabilities
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    label_map = {0: "negative", 1: "positive"}
    label = label_map[pred_idx]
    confidence = probs[pred_idx] * 100

    return f"{label} ({confidence:.1f}%)"

def main():
    parser = argparse.ArgumentParser(description="Gradio app to serve sentiment classification model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs",
        help="Path to the local directory containing the fine-tuned model files"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number to run the Gradio server on"
    )
    args = parser.parse_args()

    model_path = args.model_dir
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Specified model directory does not exist: {model_path}")

    print(f"[INFO] Loading model from: {model_path}")
    tokenizer, model = load_local_model(model_path)
    print("[INFO] Model and tokenizer loaded successfully.")

    # Wrap classification function for Gradio
    def infer(text):
        return classify_movie_review(text, tokenizer, model)

    # Build Gradio interface
    iface = gr.Interface(
        fn=infer,
        inputs=gr.Textbox(lines=3, placeholder="Nhập đoạn review phim..."),
        outputs="text",
        title="IMDb Sentiment Classifier",
        description="Ứng dụng phân loại sentiment (positive/negative) cho review phim."
    )

    # Launch Gradio server
    iface.launch(server_name="0.0.0.0", server_port=args.port)

if __name__ == "__main__":
    main()
