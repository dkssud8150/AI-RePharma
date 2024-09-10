import os
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain_openai import ChatOpenAI

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks import get_openai_callback
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, Optional, List
from llama_parse import LlamaParse
import json

import base64
from langchain_core.messages import HumanMessage


env_vars = {
    "OPENAI_API_KEY": "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",
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_API_KEY": "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",
    "LLAMA_CLOUD_API_KEY" : "llx-Fj3Mbjk18Hz9QE9z7qQd3Zy5IlrfwSbAMPAAvLUm0OUJq3LH",
}

for key, value in env_vars.items():
    os.environ[key] = value

llm = ChatOpenAI(temperature=0, model="gpt-4o")

class Relationship(BaseModel):
    entity1: str = Field(description="First entity in the relationship")
    entity1_label: str = Field(description="Label for the first entity")
    entity2: str = Field(description="Second entity in the relationship")
    entity2_label: str = Field(description="Label for the second entity")
    relationship_type: str = Field(description="Type of relationship")
    properties: dict = Field(default={}, description="Additional properties for the relationship")

# Define the model for the entire set of relationships
class Relationships(BaseModel):
    relationships: list[Relationship] = Field(description="A list of relationships")

class Description(BaseModel):
    full_description: str
    panels: Dict[str, str]
    abbreviations: str


class Figure(BaseModel):
    figure_number: str
    path: str
    page_number: int
    description: Description

class FiguresData(BaseModel):
    figures: List[Figure]

jsonparser = JsonOutputParser(pydantic_object=Relationships)
figure_parser = JsonOutputParser(pydantic_object=FiguresData)

relationship_prompt = PromptTemplate(
template="Answer the user query.\n{format_instructions}\n{query}\n",
input_variables=["query"],
partial_variables={"format_instructions": jsonparser.get_format_instructions()},
)
relationship_chain = relationship_prompt | llm | jsonparser

figure_prompt = PromptTemplate(
template="Answer the user query.\n{format_instructions}\n{query}\n",
input_variables=["query"],
partial_variables={"format_instructions": figure_parser.get_format_instructions()},
)
figure_chain = figure_prompt | llm | figure_parser

def parse_and_extract_images(pdf_directory):
    parsing_instruction = """
    You are parsing a scientific paper to extract the images and their associated metadata.
    Your task is to identify and extract each image and its associated metadata, such as page number, position, and size.
    The output should include the image file and a JSON file with the metadata for each image.
    """

    # Parse the document using LlamaParse in JSON mode
    llamaparser = LlamaParse(
        result_type="json",
        Language="EN",
        parsing_instruction=parsing_instruction,
        use_vendor_multimodal_model=False,
        vendor_multimodal_model_name="openai-gpt4o",
        vendor_multimodal_api_key=os.environ["OPENAI_API_KEY"],
    )
    json_objs = llamaparser.get_json_result(pdf_directory)

    json_path = "./Data/json_out.json"

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_objs, json_file, ensure_ascii=False, indent=4)

    print(f"JSON data saved to {json_path}")

    # Extract and save images from JSON objects
    save_directory = "./Data/Images"
    os.makedirs(save_directory, exist_ok=True)

    # Extract images from JSON and save them
    image_dicts = llamaparser.get_images(json_objs, download_path=save_directory)

    # Print the extracted images and their metadata
    for image_dict in image_dicts:
        print(f"Saved image: {image_dict['name']} with metadata: {image_dict}")

    print(f"Images and metadata saved in {save_directory}")

    return json_objs, image_dicts

def process_document(json_objs, image_dicts):

    parser = StrOutputParser()
    jsonparser = JsonOutputParser(pydantic_object=Relationships)

    abstract_system_template = """
    You are parsing a scientific paper converted to JSON format to extract the descriptions of figures, that were donwloaded as images.
    Below is a dictionary of the list of images that are present in the text.
    The dictionary tells you which page the image is present in, in "page_number": some_integer.
    From the document find the abstract text. This is typically on the first page of the document.
    """

    caption_system_template = """
    You are parsing a scientific paper converted to JSON format to extract the descriptions of figures, that were donwloaded as images.
    Below is a dictionary of the list of images that are present in the text.
    The dictionary tells you which page the image is present in, in "page_number": some_integer.

    {image_dicts}

    Figures are typically labeled with the following format: "Figure 1.", "Figure 2.", etc. 
    Each figure description often includes detailed explanations for individual panels, which are labeled as:
    (A) panel description, (B) panel description, ..., or (a) panel description, (b) panel description, ...

    Your task is to:
    1. Identify and extract each figure's full description.
    2. For each figure, extract the descriptions for all individual panels, maintaining the structure (e.g., (A), (B), etc.).
    3. Ensure that the extracted descriptions include any relevant entities (such as proteins, genes, drugs), and their relationships or interactions.
    4. Return the extracted information in a structured format, with clear labels for each figure and its corresponding panels.

    The output should maintain the context and order of the figures and their panels as presented in the paper. 
    If a figure description spans multiple paragraphs, ensure that all relevant information is captured.

    For each figure caption, you must include the "path" of the PNG file, and the "page_number" of each image.
    """

    jsonify_system_template = """
    You are tasked with converting structured figure descriptions into json format.
    The structure includes details such as the figure number (e.g. figure 1), image file path as "path" (e.g. ./Data/Images\\05a4a4b5-baee-442f-8a7c-b0101901bacc-img_p16_1.png)
    page number, a description of the figure as "description" (e.g. Protein levels were measured using ELISA), its panels as "panels", and any relevant abbreviations.
    Dont leave any of these components empty: add a random one letter string if you dont know what to put in it.
    Dont add any explanations to the output, other than the json file.
    Dont put in any ```json. Don't leave in any trailing commas.
    Make sure the entire output is a single, valid JSON object.
    """

    abstract_prompt_template = ChatPromptTemplate.from_messages(
        [("system", abstract_system_template), ("user", "{text}")]
    )

    caption_prompt_template = ChatPromptTemplate.from_messages(
        [("system", caption_system_template), ("user", "{text}")]
    )

    jsonify_prompt_template = ChatPromptTemplate.from_messages(
        [("system", jsonify_system_template), ("user", "{text}")]
    )

    abstract_chain = abstract_prompt_template | llm | parser
    caption_chain = caption_prompt_template | llm | parser
    jsonify_chain = jsonify_prompt_template | llm | parser


    abstract = abstract_chain.invoke({"text":json_objs})
    print(abstract)
    captions = caption_chain.invoke({"text":json_objs, "image_dicts":image_dicts})
    print(captions)
    json_output = jsonify_chain.invoke({"text":captions})
    print(json_output)
    json_format_output = figure_chain.invoke({"query": json_output})
    print("JSON Format Output:")
    print(json_format_output)
    print("Type of json_format_output:")
    print(type(json_format_output)) #should be dict
    print("End of JSON Format Output processing")
    return json_format_output, abstract


def image_to_base64(image_path="./Data/Images"):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def process_figures(figures, abstract):
    for figure in figures:
        image_data = image_to_base64(figure['path'])
        
        # # Prepare the panels as a formatted string
        # panels_str = "\n".join([f"{key}: {value}" for key, value in figure["panels"].items()])
        
        # Prepare the query string using the provided template
        query_str = (
            f"{abstract}\n\n"
            f"Given the images provided and the following description: {figure['description']}, "
            f"answer the query.\n"
            f"Query: Explain the relationship between entities in {figure['figure_number']}.\n"
            # f"Panels:\n{panels_str}\n"
            f"Answer: "
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": query_str},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
        
        response = llm.invoke([message])
        responses = []
        responses.append(response.content)
        print(type(responses)) #should be list
        return(responses)


def extract_and_process_relationships(responses):
    relationship = relationship_chain.invoke({"query": responses})
    print(type(relationship)) #dict
    return relationship

# if __name__ == "__main__":
#     file_name = "results.pdf"
#     pdf_directory = f"./Data/{file_name}"
    
#     # Step 1: Parse and extract images
#     json_objs, image_dicts = parse_and_extract_images(pdf_directory)
    
#     # Step 2: Process the document to get JSON output and abstract
#     json_format_output, abstract = process_document(json_objs, image_dicts)
    
#     # Step 3: Process figures to get responses
#     responses = process_figures(json_format_output["figures"], abstract)
    
#     # Step 4: Extract and process relationships
#     relationships = extract_and_process_relationships(responses)
    
#     # Print the results for verification
#     print("Abstract:")
#     print(abstract)
#     print("\nJSON Format Output:")
#     print(json_format_output)
#     print("\nResponses:")
#     print(responses)
#     print("\nRelationships:")
#     print(relationships)