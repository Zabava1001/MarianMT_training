from transformers import TrainingArguments, Trainer
from config import TRAINING_ARGS
from dataset import load_data, tokenize_data
from model import load_model, save_model


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
