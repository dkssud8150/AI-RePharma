import os

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_model(idx=None, train=False, pretrained=False, openai=False):
    ###################################################################
    # 언어 모델 설정
    ###################################################################
    model_list = ["chao1224/moleculeSTM_Graph_editing",
                  "microsoft/BioGPT",
                  "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                  "allenai/scibert_scivocab_uncased",
                  "upstage/SOLAR-10.7B-v1.0"
                  ]
    if openai:
        if 0:
            from langchain import OpenAI, LLMChain
            from langchain.prompts import PromptTemplate
            
            # opriginal_openai langchain
            llm = OpenAI(temperature=0.7)
            prompt_template = PromptTemplate(
                input_variables=["question"],
                template="Answer the following question: {question}"
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)
        else:
            # upstage openai langchain
            from langchain_upstage import ChatUpstage
            chain = ChatUpstage(api_key="up_ADtOXHjW3MUuPa5gP5o5nkW8UF15N")

        return chain

    if pretrained:
        model_name = "./biogpt"
    else:
        if idx is None:
            idx = 0
        model_name = model_list[idx]

    if train:
        dir_name = model_name.split("/")[1]
        os.makedirs(dir_name, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)

    return tokenizer, model, generator, dir_name

def extract_disease_protein_data(driver):
    data = []
    with driver.session() as session:
        cypher_query = """
        MATCH (d:Disease)-[:Disease_Protein]->(p:Protein)
        RETURN d.Name AS Disease, p.Name AS Protein
        """
        results = session.run(cypher_query)
        for record in results:
            data.append({"Disease": record["Disease"], "Protein": record["Protein"]})
    return data


def extract_data_from_neo4j(driver):
    with driver.session() as session:
        query = "MATCH (n) RETURN n.Name AS name, n.Description AS description LIMIT 100"
        result = session.run(query)
        data = [{"name": record["name"], "description": record["description"]} for record in result]
    return data

def prepare_prompts(data):
    prompts = []
    for item in data:
        prompt = f"Q: What is {item['name']}?\nA: {item['description']}\n"
        prompts.append(prompt)
    return prompts

def train_SOLAR_model_with_prompts(prompts, chain):
    training_data = "\n".join(prompts)

    response = chain.invoke(training_data)
    
    print(f"Training completed.")

def train_gpt_model_with_prompts(prompts, tokenizer, model):
    # 학습을 위해 프롬프트를 모델에 입력할 수 있는 형식으로 변환합니다.
    train_text = "\n".join(prompts)
    inputs = tokenizer(train_text, return_tensors="pt", max_length=512, truncation=True)

    # 간단한 학습 과정 구현
    model.train()
    outputs = model(**inputs, labels=inputs["input_ids"])
    
    loss = outputs.loss
    loss.backward()
    
    print(f"Training Loss: {loss.item()}")
    # 이후 학습된 모델을 저장하거나 추가 학습 가능