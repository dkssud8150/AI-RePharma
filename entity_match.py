from neo4j import GraphDatabase
import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
import re

# Environment setup
os.environ['UPSTAGE_API_KEY'] = "up_eUJl1Cy3NQq2G5QWeZm9dCKn2ruzL"
os.environ["OPENAI_API_KEY"] = "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"

# Neo4j setup
graph = Neo4jGraph(url="bolt://44.203.3.178", username="neo4j", password="trucks-triangles-toss")
driver = GraphDatabase.driver("bolt://44.203.3.178", auth=("neo4j", "trucks-triangles-toss"))

# Chat model setup
llm = ChatOpenAI(temperature=0, model="gpt-4o")

# Ensure full-text index exists or create it
def ensure_fulltext_index():
    with driver.session() as session:
        # Check if the full-text index exists
        index_check_query = """
        SHOW INDEXES YIELD name 
        WHERE name = 'entityIndex' 
        RETURN name
        """
        index_result = session.run(index_check_query)
        
        if not index_result.single():
            # Create full-text index if it doesn't exist
            create_index_query = """
            CREATE FULLTEXT INDEX entityIndex IF NOT EXISTS
            FOR (n:Drug|Gene|Disease)
            ON EACH [n.name]
            """
            session.run(create_index_query)
            print("Full-text index 'entityIndex' created.")
        else:
            print("Full-text index 'entityIndex' already exists.")

def sanitize_label(label):
    # Replace any character that is not a letter, digit, or underscore with an underscore
    return re.sub(r'[^a-zA-Z0-9_]', '_', label)

def sanitize_query(query):
    # Replace any character that is not a letter, digit, space, or underscore with a space
    return re.sub(r'[^a-zA-Z0-9_ ]', ' ', query)

def find_closest_entity_with_index(extracted_entity):
    sanitized_entity = sanitize_query(extracted_entity)  # Sanitize the entity for the query
    with driver.session() as session:
        # First, try to match the entity exactly by name
        exact_match_query = """
        MATCH (n {name: $entity})
        RETURN n
        LIMIT 1
        """
        exact_match_result = session.run(exact_match_query, entity=sanitized_entity)
        exact_record = exact_match_result.single()

        if exact_record:
            node = exact_record["n"]
            return node["name"], 1.0  # Exact match found, return score 1.0

        # Fallback to full-text search if exact match not found
        query = """
        CALL db.index.fulltext.queryNodes('entityIndex', $entity)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT 1
        """
        result = session.run(query, entity=sanitized_entity)
        record = result.single()
        
        if record:
            node = record["node"]
            node_name = node.get("Name") or node.get("name")
            return node_name, record["score"]
        
        return None, 0

def entity_matcher(extracted_entity, entity_label):
    ensure_fulltext_index()  # Ensure the full-text index exists
    closest, score = find_closest_entity_with_index(extracted_entity)

    if closest:
        print(f"Entity '{extracted_entity}' matched with '{closest}' (score: {score})")
        return closest  # Return the matched entity
    else:
        print(f"No match found for '{extracted_entity}'. Creating a new node.")
        sanitized_label = sanitize_label(entity_label)  # Sanitize the label
        with driver.session() as session:
            create_node_query = f"CREATE (n:{sanitized_label} {{name: $name}}) RETURN n"
            session.run(create_node_query, name=extracted_entity)
        return extracted_entity  # Return the created entity

def relationship_matcher(entity1, entity2, original_relationship_type, properties=None):
    with driver.session() as session:
        # Check if the :INTERACTS relationship already exists between entity1 and entity2
        relationship_query = """
        MATCH (a {name: $entity1})-[r:interacts]->(b {name: $entity2})
        RETURN r
        """
        result = session.run(relationship_query, entity1=entity1, entity2=entity2)

        if result.single():
            print(f"Relationship 'INTERACTS' between '{entity1}' and '{entity2}' already exists.")
        else:
            print(f"Would create relationship 'INTERACTS' between '{entity1}' and '{entity2}'.")

            # Prepare the properties, including the original relationship type
            properties = properties or {}
            properties['type'] = original_relationship_type  # Store the original relationship type as a property
            
            # Convert properties to a Cypher-friendly string
            properties_str = ', '.join([f'{key}: ${key}' for key in properties.keys()])
            
            # Print or execute the creation query
            print(f"""
            MATCH (a {{name: '{entity1}'}}), (b {{name: '{entity2}'}})
            CREATE (a)-[:INTERACTS {{ {properties_str} }}]->(b)
            """)
            
            # If you want to modify the database, uncomment the following:
            # session.run(f"""
            # MATCH (a {{name: $entity1}}), (b {{name: $entity2}})
            # CREATE (a)-[:INTERACTS {{ {properties_str} }}]->(b)
            # """, entity1=entity1, entity2=entity2, **properties)

def match_entities_and_relationships(json_output):
    matched_entities = {}
    matched_relationships = []

    for relationship in json_output["relationships"]:
        entity1 = entity_matcher(relationship["entity1"], relationship["entity1_label"])
        entity2 = entity_matcher(relationship["entity2"], relationship["entity2_label"])
        relationship_type = relationship["relationship_type"]

        matched_relationships.append({
            "entity1": entity1,
            "entity2": entity2,
            "relationship_type": relationship_type,
            "properties": relationship.get("properties", {})
        })

        matched_entities[entity1] = relationship["entity1"]
        matched_entities[entity2] = relationship["entity2"]

    matched = {
        "matched_entities": matched_entities,
        "matched_relationships": matched_relationships
    }

    return matched

# def process_relationships_from_json(json_data):
#     relationships = json_data["relationships"]
    
#     for relationship in relationships:
#         # Match the entities in the graph (or create them if they don't exist)
#         entity1 = entity_matcher(relationship["entity1"], relationship["entity1_label"])
#         entity2 = entity_matcher(relationship["entity2"], relationship["entity2_label"])
        
#         # Extract relationship type and properties
#         relationship_type = relationship["relationship_type"]
#         properties = relationship.get("properties", {})
        
#         # Create or validate the relationship between entities
#         relationship_matcher(entity1, entity2, relationship_type, properties)




# json_data = {
#     'relationships': [
#         {
#             'entity1': 'Doxycycline (Dox)',
#             'entity1_label': 'Treatment',
#             'entity2': 'RUNX2',
#             'entity2_label': 'Protein',
#             'relationship_type': 'Induction',
#             'properties': {
#                 'description': 'Dox treatment significantly increases the levels of FLAG-RUNX2 and endogenous RUNX2 at both Day 2 and Day 6.'
#             }
#         },
#         {
#             'entity1': 'Doxycycline (Dox)',
#             'entity1_label': 'Treatment',
#             'entity2': 'Runx2',
#             'entity2_label': 'mRNA',
#             'relationship_type': 'Induction',
#             'properties': {
#                 'description': 'Dox treatment significantly increases Runx2 mRNA levels compared to the control.'
#             }
#         },
#         {
#             'entity1': 'Estradiol (E2)',
#             'entity1_label': 'Treatment',
#             'entity2': 'Runx2',
#             'entity2_label': 'mRNA',
#             'relationship_type': 'No significant effect',
#             'properties': {
#                 'description': 'Estradiol alone does not significantly affect Runx2 mRNA levels.'
#             }
#         },
#         {
#             'entity1': 'Doxycycline (Dox) + Estradiol (E2)',
#             'entity1_label': 'Combined Treatment',
#             'entity2': 'Runx2',
#             'entity2_label': 'mRNA',
#             'relationship_type': 'Reduction',
#             'properties': {
#                 'description': 'The combination of Dox and Estradiol results in a significant reduction in Runx2 mRNA levels compared to Dox alone.'
#             }
#         },
#         {
#             'entity1': 'Doxycycline (Dox)',
#             'entity1_label': 'Treatment',
#             'entity2': 'Osteocalcin (Oc)',
#             'entity2_label': 'mRNA',
#             'relationship_type': 'Induction',
#             'properties': {
#                 'description': 'Dox treatment significantly increases Oc mRNA levels compared to the control.'
#             }
#         },
#         {
#             'entity1': 'Estradiol (E2)',
#             'entity1_label': 'Treatment',
#             'entity2': 'Osteocalcin (Oc)',
#             'entity2_label': 'mRNA',
#             'relationship_type': 'No significant effect',
#             'properties': {
#                 'description': 'Estradiol alone does not significantly affect Oc mRNA levels.'
#             }
#         },
#         {
#             'entity1': 'Doxycycline (Dox) + Estradiol (E2)',
#             'entity1_label': 'Combined Treatment',
#             'entity2': 'Osteocalcin (Oc)',
#             'entity2_label': 'mRNA',
#             'relationship_type': 'Reduction',
#             'properties': {
#                 'description': 'The combination of Dox and Estradiol results in a significant reduction in Oc mRNA levels compared to Dox alone.'
#             }
#         },
#         {
#             'entity1': 'Doxycycline (Dox)',
#             'entity1_label': 'Treatment',
#             'entity2': 'Osterix (Osx)',
#             'entity2_label': 'mRNA',
#             'relationship_type': 'Induction',
#             'properties': {
#                 'description': 'Dox treatment significantly increases Osx mRNA levels compared to the control.'
#             }
#         },
#         {
#             'entity1': 'Estradiol (E2)',
#             'entity1_label': 'Treatment',
#             'entity2': 'Osterix (Osx)',
#             'entity2_label': 'mRNA',
#             'relationship_type': 'No significant effect',
#             'properties': {
#                 'description': 'Estradiol alone does not significantly affect Osx mRNA levels.'
#             }
#         },
#         {
#             'entity1': 'Doxycycline (Dox) + Estradiol (E2)',
#             'entity1_label': 'Combined Treatment',
#             'entity2': 'Osterix (Osx)',
#             'entity2_label': 'mRNA',
#             'relationship_type': 'Reduction',
#             'properties': {
#                 'description': 'The combination of Dox and Estradiol results in a significant reduction in Osx mRNA levels compared to Dox alone.'
#             }
#         }
#     ]
# }


# # Testing
# result = match_entities_and_relationships(json_data)
# print(result)
# print(type(result))






