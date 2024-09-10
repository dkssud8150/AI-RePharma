import os

from neo4j import GraphDatabase

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import uvicorn
from asgiref.wsgi import WsgiToAsgi
from werkzeug.utils import secure_filename
from langchain_community.graphs import Neo4jGraph

###################################################################
# Custom Function
###################################################################
from article_extractor import *
from path_algorithm import get_graph_compatible_data
from entity_match import match_entities_and_relationships

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'pdf'}

# Check extension for input file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

env_vars = {
    "OPENAI_API_KEY": "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",
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_API_KEY": "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",
    "LLAMA_CLOUD_API_KEY" : "llx-Fj3Mbjk18Hz9QE9z7qQd3Zy5IlrfwSbAMPAAvLUm0OUJq3LH",
}

for key, value in env_vars.items():
    os.environ[key] = value

graph = Neo4jGraph(url="bolt://44.203.3.178", username="neo4j", password="trucks-triangles-toss")
driver = GraphDatabase.driver("bolt://44.203.3.178", auth=("neo4j", "trucks-triangles-toss"))

llm = ChatOpenAI(temperature=0, model="gpt-4o")
chat = ChatOpenAI(temperature=0, model="gpt-4o")

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


def execute_query(tx, queries):
    if isinstance(queries, list):
        for query in queries:
            try:
                tx.run(query)
                print("Relationships successfully inserted into Neo4j.")
            except Exception as e:
                print(f"Error inserting relationships into Neo4j: {e}")
                continue
    else:
        tx.run(queries)
    print("Relationships successfully inserted into Neo4j.")

def clear_database():
    delete_query = "MATCH (n) DETACH DELETE n"
    with driver.session() as session:
        session.execute_write(execute_query, delete_query)
    print("데이터베이스 초기화 완료")

def get_graph_data():
    nodes_query = "MATCH (n) RETURN n.name AS name, id(n) AS id, labels(n) AS labels LIMIT 100"
    relationships_query = """
    MATCH (n)-[r]->(m) 
    RETURN id(n) AS start_node, type(r) AS relationship, id(m) AS end_node LIMIT 100
    """
    with driver.session() as session:
        nodes = session.run(nodes_query).data()
        relationships = session.run(relationships_query).data()

    nodes_data = [{"id": node['id'], "name": node['name'], "labels": node['labels']} for node in nodes if node['name']]
    relationships_data = [{"source": rel['start_node'], "target": rel['end_node'], "relationship": rel['relationship']} for rel in relationships if rel['start_node']]

    return {"nodes": nodes_data, "links": relationships_data}

def extract_cypher_query(relationship):
    if isinstance(relationship, list):
        relationship = relationship[0]
    try:
        cypher_query_start = relationship.find("CREATE")
        queries = relationship[cypher_query_start:].strip()
        return queries
    except:
        raise ValueError("Cypher query not found in the response")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    clear_database()
    if 'question' in request.form and request.form.get('question', '') != '':
        question = request.form.get('question', '')
        query = {
            "query": "Run the algorithm",
            "drugs": question.split(',')  # Assuming drugs are comma-separated in the input
        }
        graph_compatible_data = get_graph_compatible_data(query)
        with open('relationships.json', 'w') as outfile:
            json.dump(graph_compatible_data, outfile, indent=4)
        return jsonify({'question': question, 'answer': {}, 'graphData': graph_compatible_data})

    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '' and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            upload_folder = './uploads'
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file_name)
            if not os.path.exists(file_path):
                file.save(file_path)
                print(f'File saved to {file_path}')
            else:
                print(f'{file_path} file already exists in uploads folder')

        if os.path.exists(f"{file_name.split('.', 1)[0]}_graph_data.json"):
            with open(f"{file_name.split('.', 1)[0]}_graph_data.json", 'r') as file:
                graph_data = json.load(file)
        else:
            try:
                json_objs, image_dicts = parse_and_extract_images(file_path)
                json_format_output, abstract = process_document(json_objs, image_dicts)
                responses = process_figures(json_format_output["figures"], abstract)
                relationships = extract_and_process_relationships(responses)
                matched_data = match_entities_and_relationships(relationships)
                graph_data = matched_data

                with open(f"{file_name.split('.', 1)[0]}_graph_data.json", "w") as outfile:
                    json.dump(graph_data, outfile, indent=4)
                print(f"Graph data saved to {file_name.split('.', 1)[0]}_graph_data.json")
            except Exception as e:
                print(f"Error processing document: {e}")
                return jsonify({'error': str(e)})

        print(graph_data)
        return jsonify({'question': {}, 'answer': {}, 'graphData': graph_data})

asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)