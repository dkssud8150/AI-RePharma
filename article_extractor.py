import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llama_parse import LlamaParse
import json

import base64
from langchain_core.messages import HumanMessage

parsing_instruction = """
You are parsing a scientific paper to extract the images and their associated metadata.
Your task is to identify and extract each image and its associated metadata, such as page number, position, and size.
The output should include the image file and a JSON file with the metadata for each image.
"""

# Extract images from JSON and save them
class CustomParser():
    from llama_index.core.bridge.pydantic import Field, field_validator

    base_url: str = "https://api.cloud.llamaindex.ai"
    def __init__(self, data_path, json_path, json_objs) -> None:
        os.makedirs(data_path, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_objs, json_file, ensure_ascii=False, indent=4)

    def get_images(self, json_objs, download_path, api_key):
        file_name = Path(json_objs[0]['file_path']).stem
        download_path = download_path + f"\{file_name}"
        os.makedirs(download_path, exist_ok=True)
        
        try:
            import httpx
            headers = {"Authorization": f"Bearer {api_key}"}
            images = []
            for result in json_objs:
                job_id = result["job_id"]
                for page in result["pages"]:
                    for image in page["images"]:
                        image_name = image["name"]

                        # get the full path
                        image_path = os.path.join(
                            download_path, f"{image_name}"
                        )

                        # get a valid image path
                        if not image_path.endswith(".png"):
                            if not image_path.endswith(".jpg"):
                                image_path += ".png"

                        image["path"] = image_path
                        if os.path.exists(image_path):
                            continue
                        image["job_id"] = job_id
                        image["original_pdf_path"] = result["file_path"]
                        image["page_number"] = page["page"]
                        with open(image_path, "wb") as f:
                            image_url = f"{self.base_url}/api/parsing/job/{job_id}/result/image/{image_name}"
                            f.write(httpx.get(image_url, headers=headers).content)
                        images.append(image)
            return images
        except Exception as e:
            print("Error while downloading images from the parsed result:", e)
            return []

class articleExtractor:
    # "sk-jHM43q7XKdAZLou7A8rWUd2d--Sv_-5MmNSqrTXE7sT3BlbkFJ0uY7K5f4RFR1faJgK8_jItQz-XxWlxkoY_13_a5_MA", 
    env_vars = {
        "OPENAI_API_KEY": "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", 
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": "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",
        "LLAMA_CLOUD_API_KEY" : "llx-Fj3Mbjk18Hz9QE9z7qQd3Zy5IlrfwSbAMPAAvLUm0OUJq3LH",
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    data_path = Path("./Data")
    json_path = str(data_path / "json_out.json")
    img_path = str(data_path / "Images")
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    def __init__(self, pdf_path=None):
        if pdf_path is not None:
            self.file_name = pdf_path
        else:
            self.file_name = "Estrogens Antagonize RUNX2-Mediated Osteoblast-Driven.pdf"
        
        self.pdf_directory = f"./Data/{self.file_name}"

    def extract_pdf_data(self, parsing_instruction):
        # Parse the document using LlamaParse in JSON mode
        parser = LlamaParse(
            result_type="json",
            Language="EN",
            parsing_instruction=parsing_instruction,
            use_vendor_multimodal_model=False,
            vendor_multimodal_model_name="openai-gpt-4o-mini",
            vendor_multimodal_api_key=os.environ["OPENAI_API_KEY"],
        )
        self.api_key = parser.api_key
        json_objs = parser.get_json_result(self.pdf_directory)

        os.makedirs(self.data_path,exist_ok=True)
        with open(self.json_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_objs, json_file, ensure_ascii=False, indent=4)
        self.json_objs = json_objs
        print(f"JSON data saved to {self.json_path}")

    def save_images(self):
        # Extract and save images from JSON objects
        os.makedirs(self.img_path, exist_ok=True)

        CParser = CustomParser(str(self.data_path), self.json_path, self.json_objs)
        image_dicts = CParser.get_images(self.json_objs, self.img_path, self.api_key)
        # image_dicts = parser.get_images(json_objs, download_path=save_directory)

        # Print the extracted images and their metadata
        for image_dict in image_dicts:
            print(f"Saved image: {image_dict['name']} with metadata: {image_dict}")

        self.image_dicts = image_dicts

        print(f"Images and metadata saved in {self.img_path}")

    def set_system_template(self):
        parser = StrOutputParser()

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
        The structure includes details such as the figure number (e.g. figure 1), image file path as "path" (e.g. ./Data/Images\\(pdf file name)\\img_p16_1.png) must
        page number, a description of the figure as "description" (e.g. Protein levels were measured using ELISA), its panels as "panels", and any relevant abbreviations.
        Dont leave any of these components empty: add a random one letter string if you dont know what to put in it.
        Dont add any explanations to the output, other than the json file.
        Dont put in any ```json. Don't leave in any trailing commas.
        Make sure the entire output is a single, valid JSON object.
        """

        relationship_system_template = """
        Extract all the relationships between entities from the following text in a format that is compatible with Cypher. 

        Text: "{text}"

        Relationships (in Cypher syntax):
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
        relationship_prompt_template = ChatPromptTemplate.from_messages(
            [("system", relationship_system_template), ("user", "{text}")]
        )


        abstract_chain = abstract_prompt_template | self.llm | parser
        caption_chain = caption_prompt_template | self.llm | parser
        jsonify_chain = jsonify_prompt_template | self.llm | parser
        relationship_chain = relationship_prompt_template | self.llm | parser
        
        abstract = abstract_chain.invoke({"text": self.json_objs})
        print(abstract)

        captions = caption_chain.invoke({"text": self.json_objs, "image_dicts": self.image_dicts})
        print(captions)

        json_output = jsonify_chain.invoke({"text":captions})
        print(json_output)

        json_output = json.loads(json_output)

        self.relationship_chain = relationship_chain
        self.abstract = abstract
        self.captions = captions
        self.json_output = json_output

    def process_figures(self):
        figures = self.json_output["figures"]
        responses = []
        for figure in figures:
            image_data = image_to_base64(figure['path'])
            
            # # Prepare the panels as a formatted string
            # panels_str = "\n".join([f"{key}: {value}" for key, value in figure["panels"].items()])
            
            # Prepare the query string using the provided template
            query_str = (
                f"{self.abstract}\n\n"
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
            
            response = self.llm.invoke([message])
            responses.append(response.content)
        
        relationships = []
        for response in responses:
            relationship = self.relationship_chain.invoke({"text": response})
            relationships.append(relationship)
        return relationships

#multimodal-queries of image-text
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

