import os
import pandas as pd

from neo4j import GraphDatabase
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset

from common import load_model, extract_disease_protein_data

###################################################################
# Neo4j 데이터베이스 연결 설정
###################################################################
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

driver = GraphDatabase.driver(uri, auth=(user, password))

# load model
tokenizer, model, generator, train_dir = load_model(idx=4, train=True)

# Neo4j 데이터 추출
neo4j_data = extract_disease_protein_data(driver)

# 데이터를 Q&A 형식으로 변환
qa_data = []
for item in neo4j_data:
    question = f"What proteins are associated with the disease {item['Disease']}?"
    answer = f"The protein associated with {item['Disease']} is {item['Protein']}."
    qa_data.append({"question": question, "answer": answer})

qa_data = qa_data[:200]

# 데이터셋 생성
dataset = Dataset.from_pandas(pd.DataFrame(qa_data))

tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['question'], examples['answer'], truncation=True, padding=True)

# 데이터셋 토크나이즈
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# DataCollatorForLanguageModeling 사용 mlm=False는 캐주얼 언어 모델링을 의미
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir=f"{train_dir}/results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    data_collator=data_collator
)

# 모델 훈련
trainer.train()

# 모델 저장
model.save_pretrained(f"models/{train_dir}")
tokenizer.save_pretrained(f"models/{train_dir}")