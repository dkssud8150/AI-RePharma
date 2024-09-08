from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
import os
from langchain_core.tools import tool
from langchain_core.runnables import RunnableSequence
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# # Environment setup
# os.environ['UPSTAGE_API_KEY'] = "up_eUJl1Cy3NQq2G5QWeZm9dCKn2ruzL"
# os.environ["OPENAI_API_KEY"] = "sk-proj-wDPJKrY6fv4oWsyv6nhaT3BlbkFJAI4oUTRtVIIHQqhofrCn"

# # Neo4j setup
# graph = Neo4jGraph(url="bolt://18.215.189.17:7687", username="neo4j", password="threaders-linkages-crews")
# driver = GraphDatabase.driver("bolt://18.215.189.17:7687", auth=("neo4j", "threaders-linkages-crews"))
url = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(url, auth=(user, password))
# # Chat model setup
# llm = ChatOpenAI(temperature=0, model="gpt-4o")
# chat = ChatOpenAI(temperature=0, model="gpt-4o")

def run_pagerank_on_subnetwork(tx, damping_factor=0.85, iterations=20):
    
    print("Running PageRank on subnetwork...")
    for _ in range(iterations):
        query = f"""
        MATCH (n:Disease)-[:interacts]->(m:Protein)
        WITH n, m, {damping_factor} AS dampingFactor, 0.15 AS teleport
        MATCH (r:Drug)-[:interacts]->(n), (r)-[:interacts]->(m)
        WITH n, r, m, count(*) AS inboundCount, dampingFactor, teleport
        SET n.pagerank = coalesce(n.pagerank, 1.0)
        SET n.pagerank = teleport + dampingFactor * inboundCount
        RETURN n, r, m, n.pagerank AS pagerank
        LIMIT 10
        """
        result = tx.run(query)
        result_list = [(record["n"], record["r"], record["m"], record["pagerank"]) for record in result]
        print("PageRank Results:", result_list)  # Debugging: Print results
        return result_list

def get_pagerank_results(tx):
    print("Getting PageRank results...")
    results = run_pagerank_on_subnetwork(tx)
    return {f"{n['Name']}-{d['Name']}-{m['Name']}": pagerank for n, d, m, pagerank in results}

def execute_pagerank():
    print("Executing PageRank...")
    with driver.session() as session:
        pagerank_results = session.execute_write(run_pagerank_on_subnetwork)
        print("Final PageRank Results:", pagerank_results)  # Debugging: Print final results
    return pagerank_results

@tool
def run_pagerank_with_langchain():
    """Run the PageRank algorithm on a subnetwork of nodes."""
    print("Running PageRank with LangChain...")
    results = execute_pagerank()
    formatted_results = "\n".join([f"{node['Name']}: {score:.4f}" for node, _, _, score in results])
    print("Formatted PageRank Results:", formatted_results)  # Debugging: Print formatted results
    return formatted_results

# llm_with_tools = llm.bind_tools([run_pagerank_with_langchain])

# # Define the chat prompt
# system_message = SystemMessagePromptTemplate.from_template(
#     "You are an assistant that provides PageRank analysis results."
# ) 
# human_message = HumanMessagePromptTemplate.from_template("What's the pagerank result")

# chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# # Create a chain using the chat model and prompt template
# chain = RunnableSequence(chat_prompt | llm_with_tools | run_pagerank_with_langchain)

# # Execute the chain with specific input
# response = chain.invoke({
#     "query": "What does the pagerank result tell you?"
#     }
# )

# print(response)

# print(type(response))