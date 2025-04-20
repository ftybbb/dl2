# -----------------------------
# 1. Import libraries
# -----------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader

# -----------------------------
# 2. Use GPU if available
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 3. Load and filter AG News
# -----------------------------
dataset = load_dataset("ag_news")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def filter_fn(example):
    text = example["text"]
    if not isinstance(text, str):
        return False
    if len(text.split()) < 3 or len(text) > 512:
        return False
    weird_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return (weird_chars / len(text)) <= 0.2

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# Filter + tokenize + format
dataset = dataset.filter(filter_fn)
tokenized_dataset = dataset.map(tokenize_fn, batched=False)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------------------
# 4. Load model with LoRA adapters
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

lora_config1 = LoraConfig(r=8, lora_alpha=16, target_modules=["query"], lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_CLS)
lora_config2 = LoraConfig(r=14, lora_alpha=28, target_modules=["value"], lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_CLS)

model = get_peft_model(model, lora_config1)
model = get_peft_model(model, lora_config2)
model.to(device)
model.print_trainable_parameters()

# -----------------------------
# 5. Define training args
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# -----------------------------
# 6. Train
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# -----------------------------
# 7. Evaluate
# -----------------------------
eval_results = trainer.evaluate()
print("Final Evaluation Accuracy:", eval_results["eval_accuracy"])

# -----------------------------
# 8. Check trainable parameter count
# -----------------------------
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# -----------------------------
# 9. Load and preprocess test data
# -----------------------------
with open("/home/tf2387/lora/test_unlabelled.pkl", "rb") as f:
    test_dataset = pickle.load(f)

test_dataset = Dataset.from_dict({"text": test_dataset["text"]})

def filter_test_fn(example):
    text = example["text"]
    if not isinstance(text, str):
        return False
    if len(text.split()) < 3 or len(text) > 512:
        return False
    weird_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return (weird_chars / len(text)) <= 0.2

test_dataset = test_dataset.filter(filter_test_fn)
tokenized_test_dataset = test_dataset.map(tokenize_fn, batched=True)
tokenized_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

test_dataloader = DataLoader(tokenized_test_dataset, batch_size=64)

# -----------------------------
# 10. Predict
# -----------------------------
model.eval()
all_predictions = []

with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(preds.cpu().numpy())

# -----------------------------
# 11. Save predictions
# -----------------------------
df = pd.DataFrame({
    "ID": list(range(len(all_predictions))),
    "label": all_predictions
})
df.to_csv("submission.csv", index=False)
print("✅ Predictions saved to submission.csv")
