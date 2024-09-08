import os
import pandas as pd

import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from neo4j import GraphDatabase

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import uvicorn
from asgiref.wsgi import WsgiToAsgi

###################################################################
# Custom Function
###################################################################
from common import load_model
from article_extractor import *
from langchain_neo4j_functioncall import *

app = Flask(__name__)
CORS(app)

# Environment setup
os.environ['UPSTAGE_API_KEY'] = "up_eUJl1Cy3NQq2G5QWeZm9dCKn2ruzL"
os.environ["OPENAI_API_KEY"] = "sk-proj-wDPJKrY6fv4oWsyv6nhaT3BlbkFJAI4oUTRtVIIHQqhofrCn"

###################################################################
# Neo4j 데이터베이스 연결 설정
###################################################################
url = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

# graph = Neo4jGraph(url=url, username=user, password=password)
driver = GraphDatabase.driver(url, auth=(user, password))

# Chat model setup
llm = ChatOpenAI(temperature=0, model="gpt-4o")
chat = ChatOpenAI(temperature=0, model="gpt-4o")

###################################################################
# 언어 모델 설정
###################################################################
# chain = load_model(openai=True)

def insert_relationships_into_neo4j(relationship_script):
    with driver.session() as session:
        try:
            session.run(relationship_script)
            print("Relationships successfully inserted into Neo4j.")
        except Exception as e:
            print(f"Error inserting relationships into Neo4j: {e}")

def get_graph_data_from_neo4j():
    query = """
    MATCH (n)-[r]->(m)
    RETURN n.name AS source, type(r) AS relationship, m.name AS target
    LIMIT 100
    """
    with driver.session() as session:
        results = session.run(query)
        elements = []
        for record in results:
            elements.append({"data": {"id": record["source"], "label": record["source"]}})
            elements.append({"data": {"id": record["target"], "label": record["target"]}})
            elements.append({
                "data": {
                    "id": f'{record["source"]}-{record["target"]}',
                    "source": record["source"],
                    "target": record["target"],
                    "relationship": record["relationship"]
                }
            })
        return {"elements": elements}

def extract_cypher_query(response_text):
    """
    Extract Cypher query from LLM response by removing any explanatory text.
    Assumes that the Cypher query starts after a known phrase like "Here are the relationships..."
    """
    cypher_query_start = response_text.find("CREATE")  # 'CREATE'는 일반적으로 Cypher 쿼리의 시작
    if cypher_query_start != -1:
        cypher_query = response_text[cypher_query_start:].strip()
        return cypher_query
    else:
        raise ValueError("Cypher query not found in the response")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json 
    question = data.get('question', '')  # 'question' 키로 값을 가져옴

    llm_with_tools = llm.bind_tools([run_pagerank_with_langchain])

    # Define the chat prompt
    system_message = SystemMessagePromptTemplate.from_template(
        "You are an assistant that provides PageRank analysis results."
    ) 
    human_message = HumanMessagePromptTemplate.from_template(question)
    # human_message = HumanMessagePromptTemplate.from_template("What's the pagerank result")

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # Create a chain using the chat model and prompt template
    chain = RunnableSequence(chat_prompt | llm_with_tools | run_pagerank_with_langchain)

    # Execute the chain with specific input
    response = chain.invoke({
        "query": "What does the pagerank result tell you?"
        }
    )
    answer = response


    # Process all figures
    raw_output = process_figures(json_output["figures"])
    for responses in raw_output:
        relationship = relationship_chain.invoke({"text": responses})
    print(relationship)

    relationship = relationship.strip().strip("```cypher").strip("```").strip()
    relationship = extract_cypher_query(relationship)
    insert_relationships_into_neo4j(relationship)
    graph_data = get_graph_data_from_neo4j()

    return jsonify({'question': question, 'answer': answer, 'graphData': graph_data})

asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    # app.run(debug=True)
    uvicorn.run(asgi_app, host="0.0.0.0", port=8000)