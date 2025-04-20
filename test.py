import argparse
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from torch.utils.data import DataLoader
from peft import PeftModel

# -----------------------------
# 1. Argument setup
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True, help="Path to LoRA adapter checkpoint")
parser.add_argument("--output", type=str, default="validation_misclassified_samples.csv")
args = parser.parse_args()

# -----------------------------
# 2. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 3. Load tokenizer & model
# -----------------------------
base_model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=4)
model = PeftModel.from_pretrained(base_model, args.ckpt)
model.to(device)
model.eval()

# -----------------------------
# 4. Load & preprocess dataset
# -----------------------------
dataset = load_dataset("ag_news", split="train")
id2label = dataset.features["label"].int2str

# Split off validation set (640 samples)
split = dataset.train_test_split(test_size=640, seed=42)
val_dataset = split["test"]

# Tokenization
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_val = val_dataset.map(preprocess_function, batched=True)
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
val_loader = DataLoader(tokenized_val, batch_size=64, collate_fn=data_collator)

# -----------------------------
# 5. Inference on validation set
# -----------------------------
all_preds, all_labels, all_texts = [], [], []
model.print_trainable_parameters()

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        labels = batch.pop("labels")
        input_texts = val_dataset["text"][i * batch["input_ids"].size(0):(i + 1) * batch["input_ids"].size(0)]

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_texts.extend(input_texts)

# -----------------------------
# 6. Save misclassified samples
# -----------------------------
errors = [
    {"text": t, "true_label": id2label(y), "predicted_label": id2label(p)}
    for t, y, p in zip(all_texts, all_labels, all_preds)
    if y != p
]

df_errors = pd.DataFrame(errors)
df_errors.to_csv(args.output, index=False)

# -----------------------------
# 7. Summary print
# -----------------------------
print(f"✅ Validation complete. {len(errors)} misclassified samples saved to {args.output}.")
print("\n🧪 Sample misclassified examples:")
print(df_errors.head(5).to_string(index=False))
