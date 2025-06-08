from transformers import TrainingArguments, Trainer

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config2 import TRAINING_ARGS
from src.dataset import load_data, tokenize_data
from src.model import load_model, save_model


dataset = load_data()
tokenized_dataset = tokenize_data(dataset)

model, tokenizer = load_model()

training_args = TrainingArguments(**TRAINING_ARGS)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

trainer.train()
save_model(model, tokenizer)
