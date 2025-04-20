# -----------------------------
# 1. Import libraries
# -----------------------------
import os
import pandas as pd
import torch
from transformers import RobertaModel, RobertaTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, Dataset, ClassLabel
import pickle
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
import time
import argparse

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
        base_model = 'roberta-base'

        dataset = load_dataset('ag_news', split='train')
        tokenizer = RobertaTokenizer.from_pretrained(base_model)

        def preprocess(examples):
            tokenized = tokenizer(examples['text'], truncation=True, padding=True)
            return tokenized

        tokenized_dataset = dataset.map(preprocess, batched=True,  remove_columns=["text"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

        # Extract the number of classess and their names
        num_labels = dataset.features['label'].num_classes
        class_names = dataset.features["label"].names
        print(f"number of labels: {num_labels}")
        print(f"the labels: {class_names}")

        # Create an id2label mapping
        # We will need this for our classifier.
        id2label = {i: label for i, label in enumerate(class_names)}

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

        # -----------------------------
        # 4. Load RoBERTa model with LoRA adapters
        # -----------------------------
        model = RobertaForSequenceClassification.from_pretrained(
            base_model,
            id2label=id2label)
        model.to(device)
        
        # Split the original training set
        split_datasets = tokenized_dataset.train_test_split(test_size=640, seed=42)
        train_dataset = split_datasets['train']
        eval_dataset = split_datasets['test']
        
        
        # PEFT Config
        peft_config = LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.05,
            bias = 'none',
            target_modules = ['query'],
            task_type="SEQ_CLS",
        )
        peft_model = get_peft_model(model, peft_config)
        peft_model.to(device)
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
        
        print('PEFT Model:')
        peft_model.print_trainable_parameters()
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        assert trainable_params <= 1000000, "Too many trainable parameters"
        
        

        # -----------------------------
        # 5. Define training arguments
        # -----------------------------
        # To track evaluation accuracy during training
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            # Calculate accuracy
            accuracy = accuracy_score(labels, preds)
            return {
                'accuracy': accuracy
            }
        output_dir = f"./results/{date_time}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            report_to=None,
            eval_strategy='steps',
            logging_steps=100,
            learning_rate=5e-4,
            num_train_epochs=1,
            max_steps=1200,
            use_cpu=False,
            dataloader_num_workers=4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            optim="sgd",
            gradient_checkpointing=False,
            gradient_checkpointing_kwargs={'use_reentrant':True}
        )
        # os.makedirs(output_dir, exist_ok=True)
        # with open(os.path.join(output_dir, "configs.json"), "w") as f:
        #     json.dump(vars(args), f)

        def get_trainer(model):
            return  Trainer(
                model=model,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )

        # -----------------------------
        # 6. Train the model
        # -----------------------------
        peft_lora_finetuning_trainer = get_trainer(peft_model)
        result = peft_lora_finetuning_trainer.train()



        def evaluate_model(inference_model, dataset, labelled=True, batch_size=8, data_collator=None):
            """
            Evaluate a PEFT model on a dataset.

            Args:
                inference_model: The model to evaluate.
                dataset: The dataset (Hugging Face Dataset) to run inference on.
                labelled (bool): If True, the dataset includes labels and metrics will be computed.
                                If False, only predictions will be returned.
                batch_size (int): Batch size for inference.
                data_collator: Function to collate batches. If None, the default collate_fn is used.

            Returns:
                If labelled is True, returns a tuple (metrics, predictions)
                If labelled is False, returns the predictions.
            """
            # Create the DataLoader
            eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            inference_model.to(device)
            inference_model.eval()

            all_predictions = []
            if labelled:
                metric = evaluate.load('accuracy')

            # Loop over the DataLoader
            for batch in tqdm(eval_dataloader):
                # Move each tensor in the batch to the device
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = inference_model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                all_predictions.append(predictions.cpu())

                if labelled:
                    # Expecting that labels are provided under the "labels" key.
                    references = batch["labels"]
                    metric.add_batch(
                        predictions=predictions.cpu().numpy(),
                        references=references.cpu().numpy()
                    )

            # Concatenate predictions from all batches
            all_predictions = torch.cat(all_predictions, dim=0)

            if labelled:
                eval_metric = metric.compute()
                print("Evaluation Metric:", eval_metric)
                return eval_metric, all_predictions
            else:
                return all_predictions
        _, _ = evaluate_model(peft_model, eval_dataset, True, 8, data_collator)
        unlabelled_dataset = pd.read_pickle("test_unlabelled.pkl")
        test_dataset = unlabelled_dataset.map(preprocess, batched=True, remove_columns=["text"])
        
        # Run inference and save predictions
        preds = evaluate_model(peft_model, test_dataset, False, 8, data_collator)
        df_output = pd.DataFrame({
            'ID': range(len(preds)),
            'Label': preds.numpy()  # or preds.tolist()
        })
        df_output.to_csv(os.path.join(output_dir,"inference_output.csv"), index=False)
        print("Inference complete. Predictions saved to inference_output.csv")

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