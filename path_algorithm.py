import json
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
import os
from langchain_core.tools import tool
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

os.environ['UPSTAGE_API_KEY'] = "up_eUJl1Cy3NQq2G5QWeZm9dCKn2ruzL"
os.environ["OPENAI_API_KEY"] = "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"

graph = Neo4jGraph(url="bolt://44.203.3.178", username="neo4j", password="trucks-triangles-toss")
driver = GraphDatabase.driver("bolt://44.203.3.178", auth=("neo4j", "trucks-triangles-toss"))

# Chat model setup
llm = ChatOpenAI(temperature=0, model="gpt-4o")

###########################Output parsing############################

format_system_template = '''
You are tasked with formatting the paths found between drugs, proteins, and diseases into JSON format suitable for visualization. 
The paths are provided as structured data. Format them as follows:

    """
    "paths":
        "path_id": 1,
        "start_node": "start_node_name",
        "end_node": "end_node_name",
        "path_size": path_size,
        "nodes": "id": "node_id", "type": "node_type", "name": "node_name"
        "edges": "source": "start_node_id", "target": "end_node_id", "relationship": "relationship_type", "pubmed": "pubmed_id"
        ...
    """

Paths: {paths}

Ensure that the output is valid JSON and ready for use in visualization tools.
Don't include ```json or any other extra comment to your answer.
'''

format_prompt_template = ChatPromptTemplate.from_messages(
    [("system", format_system_template), ("user", "{paths}")]
)
parser = StrOutputParser()
format_chain = format_prompt_template | llm | parser

class Node(BaseModel):
    id: int = Field(description="Unique identifier for the node")
    name: str = Field(description="Name of the node")
    labels: List[str] = Field(description="Labels associated with the node")

class Link(BaseModel):
    source: int = Field(description="ID of the source node")
    target: int = Field(description="ID of the target node")
    relationship: str = Field(description="Type of relationship between nodes")

class GraphData(BaseModel):
    nodes: List[Node] = Field(description="List of nodes in the graph")
    links: List[Link] = Field(description="List of links between nodes")

jsonparser = JsonOutputParser(pydantic_object=GraphData)
graph_compatible_prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": jsonparser.get_format_instructions()},
)
graph_compatible_chain = graph_compatible_prompt | llm | jsonparser

##############################################################################

# Define the tool without helper functions
@tool
def run_paths_with_langchain(drugs: list):
    """
    Takes a list of drugs and runs a cypher query.

    Args:
        drugs: a list of drugs
    """
    print("Running Cypher query to find paths between drugs...")

    formatted_entities = ", ".join([f"'{drug}'" for drug in drugs])
    
    cypher_query = f"""
    WITH [{formatted_entities}] AS items
    UNWIND RANGE(0, SIZE(items) - 2) AS i
    UNWIND RANGE(i + 1, SIZE(items) - 1) AS j
    WITH items[i] AS item1, items[j] AS item2
    MATCH path = shortestPath((n1)-[*]-(n2))
    WHERE n1.Name = item1 AND n2.Name = item2
    RETURN path, nodes(path) AS nodes, relationships(path) AS relationships
    """
    
    def execute_query(tx):
        result = tx.run(cypher_query)
        paths = [{"path": record["path"], "nodes": record["nodes"], "relationships": record["relationships"]} for record in result]
        return paths
    
    with driver.session() as session:
        results = session.execute_read(execute_query)
    
    if not results:
        return "No paths found between the provided drugs."

    return results  # Return the results directly

tools = [run_paths_with_langchain]
llm_with_tools = llm.bind_tools(tools)

def process_drug_paths(query, llm_with_tools, format_chain):
    human_message_content = f"The drugs you want to explore are: {query['drugs']}. Run the query: {query['query']}"
    messages = [HumanMessage(content=human_message_content)]

    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    formatted_outputs = []

    # Handle any tool calls
    for tool_call in ai_msg.additional_kwargs.get('tool_calls', []):
        tool_name = tool_call['function']['name']
        tool_args = json.loads(tool_call['function']['arguments'])  # Safe JSON parsing
        
        # Execute the selected tool using the invoke method
        selected_tool = {"run_paths_with_langchain": run_paths_with_langchain}[tool_name]
        
        # Ensure that the function is called with correct arguments (drugs list)
        tool_output = selected_tool({"drugs": tool_args["drugs"]})

        formatted_output = format_chain.invoke({"paths": tool_output})  # Using the formatting chain
        
        # Append the formatted output back as a ToolMessage
        messages.append(ToolMessage(content=formatted_output, tool_call_id=tool_call['id']))
        formatted_outputs.append(formatted_output)

    # Display the final message sequence
    for msg in messages:
        print(msg.content)

    return formatted_outputs

def get_graph_compatible_data(query):
    paths = process_drug_paths(query, llm_with_tools, format_chain)
    graph_compatible_data = graph_compatible_chain.invoke({"query": paths})
    print(json.dumps(graph_compatible_data, indent=2))

    return graph_compatible_data

# Testing
# paths = process_drug_paths(query, llm_with_tools, format_chain)
# graph_compatible_data = graph_compatible_chain.invoke({"query": paths})


# print(f"Type of graph_compatible_data: {type(graph_compatible_data)}")
# print(json.dumps(graph_compatible_data, indent=2))

# Test the get_graph_compatible_data function
if __name__ == "__main__":
    test_query = {
        "query": "Run the algorithm",
        "drugs": ["Osteoporosis", "zoledronic acid", "Alendronate", "Pentoxifylline"]
    }
    
    result = get_graph_compatible_data(test_query)
    print("Test Result:")
    print(json.dumps(result, indent=2))
