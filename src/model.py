from transformers import MarianMTModel, MarianTokenizer
from config import MODEL_PATH, SAVE_PATH, device


def load_model():
    tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
    model = MarianMTModel.from_pretrained(MODEL_PATH).to(device)

    print(f"Модель загружена на: {device}")

    return model, tokenizer


def save_model(model, tokenizer):
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
