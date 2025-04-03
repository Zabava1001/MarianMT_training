import os
from transformers import MarianMTModel, MarianTokenizer


model_name = "Helsinki-NLP/opus-mt-ru-en"
save_path = "./model"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

os.makedirs(save_path, exist_ok=True)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Модель сохранена в {save_path}")
