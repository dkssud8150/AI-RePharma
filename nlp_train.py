import torch
from neo4j import GraphDatabase

###################################################################
# Neo4j 데이터베이스 연결 설정
###################################################################
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

driver = GraphDatabase.driver(uri, auth=(user, password))

def extract_data_from_neo4j():
    with driver.session() as session:
        query = "MATCH (n) RETURN n.XID AS xid, n.Name AS name, n.ID AS id, n.NameSpace AS namespace LIMIT 100"
        result = session.run(query)
        data = []
        for record in result:
            data.append({
                "xid": record["xid"],
                "name": record["name"],
                "id": record["id"],
                "namespace": record["namespace"]
            })
    return data

def prepare_ner_data(data):
    ner_data = []
    for item in data:
        # 여러 필드를 결합하여 문장을 구성
        sentence = f"{item['name']} is identified by ID {item['id']} in the namespace {item['namespace']}."
        words = sentence.split()  # 문장을 단어로 분리
        labels = ['O'] * len(words)  # 기본 레이블은 모두 'O' (엔티티 외부)

        # 'Name', 'ID', 'NameSpace'와 같은 중요한 정보를 NER 태그로 지정
        for i, word in enumerate(words):
            if word == item['name']:
                labels[i] = 'B-NAME'  # Name 엔티티의 시작
            elif word == item['id']:
                labels[i] = 'B-ID'  # ID 엔티티의 시작
            elif word == item['namespace']:
                labels[i] = 'B-NAMESPACE'  # NameSpace 엔티티의 시작

        ner_data.append({"words": words, "labels": labels})
    
    return ner_data

from transformers import AutoTokenizer

# BERT 모델을 위한 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

label_map = {
    "O": 0,
    "B-NAME": 1,
    "B-ID": 2,
    "B-NAMESPACE": 3
}

def tokenize_and_align_labels(ner_data, tokenizer):
    tokenized_inputs = tokenizer(
        [item['words'] for item in ner_data],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    labels = []
    for i, label in enumerate([item['labels'] for item in ner_data]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[label[word_idx]])  # 문자열을 정수로 변환
            else:
                label_ids.append(-100)  # 하나의 단어가 여러 토큰으로 분리될 때, 첫 번째 토큰에만 라벨 적용
            previous_word_idx = word_idx

        # 레이블이 없는 경우 빈 배열이 아닌 -100으로 채워진 배열을 추가합니다.
        if len(label_ids) == 0:
            label_ids = [-100] * len(word_ids)

        labels.append(label_ids)

    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    return tokenized_inputs

from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments

# NER을 위한 BERT 모델 로드 (크기 불일치 무시)
model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english",
    num_labels=3,  # NER 태그 수에 맞게 설정
    ignore_mismatched_sizes=True  # 크기 불일치 무시
)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="nlp/results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)


if __name__ == '__main__':
    # 1. Neo4j에서 데이터 추출
    data = extract_data_from_neo4j()

    # 2. NER 데이터 준비
    ner_data = prepare_ner_data(data)

    # 3. 토큰화 및 레이블 정렬
    tokenized_inputs = tokenize_and_align_labels(ner_data, tokenizer)

    from datasets import Dataset
    train_dataset = Dataset.from_dict(tokenized_inputs)

    # Trainer 객체 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 4. 모델 학습
    trainer.train()

    # 5. 평가 및 저장
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    model.save_pretrained("./my_ner_model")
    tokenizer.save_pretrained("./my_ner_model")
