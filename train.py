import os
os.environ["USE_TF"] = "NO"
os.environ["USE_KERAS"] = "NO"

import torch
import logging
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

logging.getLogger("transformers").setLevel(logging.ERROR)

class FinancialReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model(texts, labels):
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = f'./models/finbert_{current_time}'
    model_name = "kwoncho/KoFinBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

    train_dataset = FinancialReportDataset(texts, labels, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,        
        per_device_train_batch_size=2,
        logging_steps=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    print("🚀 학습 시작...")
    trainer.train()
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ 학습 완료! 모델이 {save_path}에 저장되었습니다.")

if __name__ == "__main__":
    sample_texts = ["TC본더는 HBM 생산의 핵심 장비이다.", "당분간 보수적인 접근이 필요하다."]
    sample_labels = [0, 1]
    train_model(sample_texts, sample_labels)