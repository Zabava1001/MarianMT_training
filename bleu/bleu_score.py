from transformers import MarianMTModel, MarianTokenizer
from src.config import SAVE_PATH, MAX_LENGTH, SAMPLE_SIZE, BATCH_SIZE_TRAIN, BLUE_PATH
from src.dataset import load_data

import evaluate
import torch


def generate_translations_batched(texts, model, tokenizer, batch_size=16, max_length=128, device="cuda"):
    model.to(device)
    model.eval()
    translations = []

    for i in range(0, len(texts), batch_size):
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

model = MarianMTModel.from_pretrained(model_path, local_files_only=True)
tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)

dataset = load_data(path=BLUE_PATH)

print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")

bleu = evaluate.load("bleu")

sample_size = SAMPLE_SIZE
random_sample = dataset['test'].shuffle(seed=42).select([i for i in range(sample_size)])
print(f"Sample size: {len(random_sample)}")

reference_sample = random_sample['khakas']

translations_decoded = generate_translations_batched(
    list(random_sample['russian']),
    model,
    tokenizer,
    batch_size=BATCH_SIZE_TRAIN,
    max_length=MAX_LENGTH,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

for i in range(5):
    print(f"Source: {random_sample[i]['russian']}")
    print(f"Translate: {translations_decoded[i]}")
    print(f"Target: {random_sample[i]['khakas']}")
    print("-" * 50)

results_sample = bleu.compute(predictions=translations_decoded, references=[[ref] for ref in reference_sample])
print(f"BLEU score on sample: {results_sample['bleu']}")
