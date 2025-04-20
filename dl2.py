# -----------------------------
# 1. Import libraries
# -----------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import pickle
import os
import argparse
import time
import json
import csv
import torch.nn as nn
# -----------------------------
# 2. Use GPU if available
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



if __name__ == "__main__":
    date_time = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="train")
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3) #2e-4
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--nonlinear', action='store_true')
    parser.add_argument('--target', type=str, default="query,value")
    parser.add_argument('--scheduler', type=str, default="constant")
    args = parser.parse_args()
    
    if args.task == "train":
            # -----------------------------
        # -----------------------------
        # 3. Load and preprocess AGNEWS dataset
        # -----------------------------
        dataset = load_dataset("ag_news")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        def analyze_dataset(dataset, tokenizer):
            lengths = [len(tokenizer(text)['input_ids']) for text in dataset['text']]
            print(f"Min length: {min(lengths)}")
            print(f"Max length: {max(lengths)}")
            print(f"Average length: {sum(lengths) / len(lengths)}")
            print(f'90% length: {sorted(lengths)[int(len(lengths) * 0.9)]}')
            print(f'99% length: {sorted(lengths)[int(len(lengths) * 0.95)]}')
            print(f"Median length: {sorted(lengths)[len(lengths) // 2]}")
            

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


        # -----------------------------
        # 4. Load RoBERTa model with LoRA adapters
        # -----------------------------
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)
        # target_modules = args.target.split(",")
        # lora_config1 = LoraConfig(
        #     r=8,
        #     lora_alpha=16,
        #     target_modules=["query"],
        #     lora_dropout=0.1,
        #     bias="none",
        #     task_type=TaskType.SEQ_CLS
        # )
        # lora_config2 = LoraConfig(
        #     r=14,
        #     lora_alpha=28,
        #     target_modules=["value"],
        #     lora_dropout=0.1,
        #     bias="none",
        #     task_type=TaskType.SEQ_CLS
        # )

        # model = get_peft_model(model, lora_config1)
        # model = get_peft_model(model, lora_config2)
        
        # one config
        lora_config = LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            target_modules=args.target.split(","),
            lora_dropout=args.dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, lora_config)
        model.to(device)
        if args.nonlinear:
            hidden_dim = model.classifier.dense.in_features
            print(f"Hidden dimension: {hidden_dim}")
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 4)
            )
        
        
        model.print_trainable_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params <= 1000000, "Too many trainable parameters"
        
        

        # -----------------------------
        # 5. Define training arguments
        # -----------------------------
        output_dir = f"./results/{date_time}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
            learning_rate=args.lr,
            lr_scheduler_type=args.scheduler,
            warmup_ratio=0.1,
            label_smoothing_factor=0.1,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=64,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            logging_dir="./logs",
            report_to="none"
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "configs.json"), "w") as f:
            json.dump(vars(args), f)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {"accuracy": accuracy_score(labels, predictions)}

        # -----------------------------
        # 6. Train the model
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
        # 7. Evaluate the model
        # -----------------------------
        eval_results = trainer.evaluate()
        print("Final Evaluation Accuracy:", eval_results["eval_accuracy"])

        # -----------------------------
        # 8. Check trainable parameter count
        # -----------------------------
        
        print(f"Trainable parameters: {trainable_params}")
        
        stat_file = os.path.join('results', 'stats.csv')
        file_exist = os.path.exists(stat_file)
        with open(stat_file, 'a', newline='') as f:
            res = [date_time, eval_results["eval_accuracy"], trainable_params]
            writer = csv.writer(f)
            if not file_exist: 
                writer.writerow(['path', 'eval_accuracy', 'trainable_params'])
            writer.writerow(res)

    if args.task == "test":
        # -----------------------------
        from datasets import Dataset
        from torch.utils.data import DataLoader
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)
        model = PeftModel.from_pretrained(base_model, args.ckpt)
        model.to(device)
        model.eval()

        # Load dataset object
        with open("test_unlabelled.pkl", "rb") as f:
            test_dataset = pickle.load(f)

        # Convert to HuggingFace Dataset (already is, but this helps formatting)
        test_dataset = Dataset.from_dict({"text": test_dataset["text"]})

        # Tokenize function
        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

        # Apply tokenizer
        tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
        tokenized_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Create PyTorch DataLoader for batching
        test_dataloader = DataLoader(tokenized_test_dataset, batch_size=64)

        # Prediction loop
        model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(preds.cpu().numpy())
                
        # -----------------------------
        # 10. Save predictions to CSV
        # -----------------------------
        df = pd.DataFrame({
            "ID": list(range(len(all_predictions))),   # ID ✅
            "label": all_predictions
        })
        df.to_csv("submission.csv", index=False)
        print("✅ Batched predictions complete. Saved to submission.csv.")