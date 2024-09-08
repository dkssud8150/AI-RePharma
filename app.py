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
os.environ["OPENAI_API_KEY"] = "sk-jHM43q7XKdAZLou7A8rWUd2d--Sv_-5MmNSqrTXE7sT3BlbkFJ0uY7K5f4RFR1faJgK8_jItQz-XxWlxkoY_13_a5_MA" # "sk-proj-wDPJKrY6fv4oWsyv6nhaT3BlbkFJAI4oUTRtVIIHQqhofrCn"

###################################################################
# Neo4j 데이터베이스 연결 설정
###################################################################
# class Neo4jConnection:
#     def __init__(self, url, user, password) -> None:
#         self.driver = GraphDatabase.driver(url, auth=(user, password))
#     def close(self):
#         self.driver.close()
#     def execute_query(self, query):
#         with self.driver.session() as session:
#             try:
#                 session.run(query)
#                 print("Relationships successfully inserted into Neo4j.")
#             except Exception as e:
#                 print(f"Error inserting relationships into Neo4j: {e}")


def execute_query(query):
    with driver.session() as session:
        try:
            session.run(query)
            print("Relationships successfully inserted into Neo4j.")
        except Exception as e:
            print(f"Error inserting relationships into Neo4j: {e}")

def clear_database():
    delete_query = "MATCH (n) DETACH DELETE n"
    execute_query(delete_query)
    print("데이터베이스 초기화 완료")

clear_database()

url = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(url, auth=(user, password))

# Chat model setup
llm = ChatOpenAI(temperature=0, model="gpt-4o")
chat = ChatOpenAI(temperature=0, model="gpt-4o")

###################################################################
# 언어 모델 설정
###################################################################
# chain = load_model(openai=True)

def get_graph_data():
    """ query
    // Create nodes
    CREATE (NeMCO:CellType {name: 'Newborn Mouse Calvarial Osteoblasts'})
    CREATE (RUNX2:Gene {name: 'RUNX2'})
    CREATE (FLAG_RUNX2:Protein {name: 'FLAG-RUNX2'})
    CREATE (Estradiol:Compound {name: 'Estradiol'})
    CREATE (Doxycycline:Compound {name: 'Doxycycline'})
    CREATE (Control:Condition {name: 'Control'})
    CREATE (Runx2:Gene {name: 'Runx2'})
    CREATE (Osteocalcin:Gene {name: 'Osteocalcin'})
    CREATE (Osterix:Gene {name: 'Osterix'})
    CREATE (Oc:Gene {name: 'Oc'})
    CREATE (Osx:Gene {name: 'Osx'})

    // Create relationships
    CREATE (NeMCO)-[:TRANSDUCED_WITH]->(RUNX2)
    CREATE (NeMCO)-[:TREATED_WITH]->(Doxycycline)
    CREATE (NeMCO)-[:TREATED_WITH]->(Control)
    CREATE (NeMCO)-[:EXPRESSION_OF]->(RUNX2)
    CREATE (NeMCO)-[:EXPRESSION_OF]->(Runx2)
    CREATE (NeMCO)-[:EXPRESSION_OF]->(Osteocalcin)
    CREATE (NeMCO)-[:EXPRESSION_OF]->(Osterix)

    // Induction relationships
    CREATE (Doxycycline)-[:INDUCES]->(RUNX2)
    CREATE (Doxycycline)-[:INCREASES]->(Runx2)
    CREATE (Doxycycline)-[:ENHANCES]->(Osteocalcin)
    CREATE (Doxycycline)-[:INCREASES]->(Osterix)

    // Inhibition relationships
    CREATE (Estradiol)-[:INHIBITS]->(Runx2)
    CREATE (Estradiol)-[:REDUCES]->(Osteocalcin)
    CREATE (Estradiol)-[:REDUCES]->(Osterix)
    CREATE (Estradiol)-[:ANTAGONIZES]->(RUNX2)

    // Regulatory relationships
    CREATE (RUNX2)-[:PROMOTES]->(Osteocalcin)
    CREATE (RUNX2)-[:PROMOTES]->(Osterix)
    CREATE (Estradiol)-[:INHIBITS]->(RUNX2)
    """

    nodes_query = "MATCH (n) RETURN n.name AS name, id(n) AS id, labels(n) AS labels LIMIT 100"
    relationships_query = """
    MATCH (n)-[r]->(m) 
    RETURN id(n) AS start_node, type(r) AS relationship, id(m) AS end_node LIMIT 100
    """

    # 노드와 관계 조회
    with driver.session() as session:
        nodes = session.run(nodes_query)
        nodes = nodes.data()
        relationships = session.run(relationships_query)
        relationships = relationships.data()
    # nodes = connection.execute_query(nodes_query)
    # relationships = connection.execute_query(relationships_query)

    # 노드를 JSON 형식으로 변환
    nodes_data = []
    for node in nodes:
        if not node['name']: continue
        nodes_data.append({
            "id": node['id'],
            "name": node['name'],
            "labels": node['labels']
        })

    # 관계를 JSON 형식으로 변환
    relationships_data = []
    for rel in relationships:
        if not rel['start_node']: continue
        relationships_data.append({
            "source": rel['start_node'],
            "target": rel['end_node'],
            "relationship": rel['relationship']
        })

    # 그래프 데이터를 합쳐서 반환
    graph_data = {
        "nodes": nodes_data,
        "links": relationships_data
    }

    return graph_data

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
    # human_message = HumanMessagePromptTemplate.from_template("What's the pagerank result")
    human_message = HumanMessagePromptTemplate.from_template(question)

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # Create a chain using the chat model and prompt template
    chain = RunnableSequence(chat_prompt | llm_with_tools | run_pagerank_with_langchain)

    # Execute the chain with specific input
    response = chain.invoke({
        "query": "What does the pagerank result tell you?"
        }
    )
    answer = response

    if os.path.exists("graph_data.json"):
        with open('graph_data.json', 'r') as file:
            graph_data = json.load(file)
    else:
        # Process all figures
        raw_output = process_figures(json_output["figures"])
        for responses in raw_output:
            relationship = relationship_chain.invoke({"text": responses})
        print(relationship)

        relationship = relationship.strip().strip("```cypher").strip("```").strip()
        relationship = extract_cypher_query(relationship)
        execute_query(relationship)
        graph_data = get_graph_data()
        
        # JSON 파일로 저장
        with open("graph_data.json", "w") as outfile:
            json.dump(graph_data, outfile, indent=4)

    print(graph_data)

    return jsonify({'question': question, 'answer': answer, 'graphData': graph_data})

asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    # app.run(debug=True)
    uvicorn.run(asgi_app, host="0.0.0.0", port=8000)