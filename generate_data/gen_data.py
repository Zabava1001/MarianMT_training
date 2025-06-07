from transformers import MarianMTModel, MarianTokenizer
from src.config2 import SAVE_PATH, MAX_LENGTH, BLUE_PATH
from src.dataset import load_data

import torch


def generate_translations_batched(texts, model, tokenizer, batch_size=16, max_length=128, device="cuda"):
    model.to(device)
    model.eval()
    translations = []

    for i in range(0, len(texts), batch_size):
        print(i)
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        translations.extend(decoded)

    return translations


model_path = SAVE_PATH

print("Загрузка модели")
model = MarianMTModel.from_pretrained(model_path, local_files_only=True)
tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)

print("Загрузка датасета")
dataset = load_data(path=BLUE_PATH)

sample_size = 26100
random_sample = dataset['test'].shuffle(seed=42).select([i for i in range(sample_size)])

reference_sample = random_sample['russian']
translations_decoded = generate_translations_batched(
    list(random_sample['khakas']),
    model,
    tokenizer,
    batch_size=16,
    max_length=MAX_LENGTH,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

import pandas as pd

df = pd.DataFrame({
    "khakas": random_sample["khakas"],
    "russian": translations_decoded
})

df.to_excel("synthetic_augmented_dataset.xlsx", index=False, header=False)

print("Файл сохранён без заголовков: synthetic_augmented_dataset.xlsx")
