from transformers import MarianMTModel, MarianTokenizer
from src.config import SAVE_PATH, MAX_LENGTH, SAMPLE_SIZE
from src.dataset import load_data

import evaluate
import torch


model_path = SAVE_PATH

model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_path)

dataset = load_data()

print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")

bleu = evaluate.load("bleu")

sample_size = SAMPLE_SIZE
random_sample = dataset['test'].shuffle(seed=42).select([i for i in range(sample_size)])
print(f"Sample size: {len(random_sample)}")

inputs = tokenizer(list(random_sample['russian']), return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    translations = model.generate(inputs['input_ids'], max_length=MAX_LENGTH)

translations_decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in translations]
reference_sample = random_sample['khakas']

for i in range(5):
    print(f"Source: {random_sample[i]['russian']}")
    print(f"Translate: {translations_decoded[i]}")
    print(f"Target: {random_sample[i]['khakas']}")
    print("-" * 50)

results_sample = bleu.compute(predictions=translations_decoded, references=[[ref] for ref in reference_sample])
print(f"BLEU score on sample: {results_sample['bleu']}")
