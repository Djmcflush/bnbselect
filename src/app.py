from segmentmodel import catpion_model
from typing import List
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from airbnb_model import airbnb_output_parser, AirbnbListing
import requests
from listings import (
    read_listings_from_pinecone_db,
    write_to_pinecone_db,
    fetch_top_k_from_pinecone_db,
    embed_text_with_pinecone,
)
import os

OPEN_AI_API_KEY = os.environ.get('OPEN_AI_API_KEY')
PINECONE_AIRBNB_EMDEDDING_INDEX = os.environ.get('PINECONE_EMBEDDING_INDEX_NAME')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

caption_model, processor = catpion_model()
read_index = os.environ.get("READ_INDEX")
write_index = os.environ.get("WRITE_INDEX")

def prediction_pipeline(image, iterate_over_airbnb_fields=False):
    captions = []
    if not iterate_over_airbnb_fields:
        return caption_prediction(image, field="Describe All the objects in this image")
    captions = []
    for field in AirbnbListing.__fields__:
        caption = caption_prediction(image, field.description)
        captions.append(caption)
    return " ".join(captions)


def caption_prediction(image, field):
    inputs = processor(raw_image, field, return_tensors="pt")
    return catpion_model.generate(**inputs)


def airbnb_prediction(query):
    _input = prompt.format_prompt(query=query)
    parser = airbnb_output_parser()
    prompt = PromptTemplate(
        template="Describe all the features and amenities inside of this description.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    model_name = "chatgpt-3.5"
    temperature = 0.0
    model = OpenAI(model_name=model_name, temperature=temperature)

    output = model(_input.to_string())
    return parser.parse(output)


def disambiguate_query(query):
    _input = prompt.format_prompt(query=query)
    prompt = PromptTemplate(
        template="Describe all the features and amenities inside of this description.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
    )
    model_name = "chatgpt-3.5"
    temperature = 0.0
    model = OpenAI(model_name=model_name, temperature=temperature)

    output = model(_input.to_string())
    return output


def captioning_machine(images):
    captions = []
    for image in images:  # for each image in a listing
        caption = prediction_pipeline(image)
        captions.append(caption)
    return airbnb_prediction(captions)


# Streamlit app
def disambiguate(description, caption):
    # TODO
    # do something to the data to "disambiguate it"
    # most likely just prompt it like this:
    return disambiguate_query(description + caption)


def create_final_text(descriptions, captions, disambiguation):
    # we need to update this!!!! TODO
    return descriptions + captions + disambiguation


def process_listing(listing):
    raw_images = listing.get("images")
    images = []
    for image in images:
        im = Image.open(requests.get(image, stream=True).raw)
        images.append(im)
    
    description = listing.get("description") #still need to figure out description format #TODO
    captions = captioning_machine(images)
    formated_description = airbnb_prediction(description)
    disambiguated = disambiguate(formated_description, captions)
    # come up with some final data format???
    final_data_format = create_final_text(formated_description, captions, disambiguated)
    embedding = embed_text_with_pinecone(final_data_format)

    write_to_pinecone_db(PINECONE_AIRBNB_EMDEDDING_INDEX, embedding)


def main():
    listings = read_listings_from_pinecone_db("Index name")
    for listing in listings:
        process_listing(listing)
