from transformers import MarianMTModel, MarianTokenizer
from config2 import SAVE_PATH
import torch


model_path = SAVE_PATH
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def translate(text):
    text = f">>khk<< {text}"

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        translated_tokens = model.generate(**inputs)

    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_text


russian_text = "Чалахай иртеннең! Хакас тілі хайди даа кірерге кирек!"
khakas_translation = translate(russian_text)
print(f"Перевод: {khakas_translation}")
