from segmentmodel import catpion_model
from typing import List
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from json import dumps
from langchain.docstore.document import Document

from airbnb_model import (
    airbnb_output_parser,
    final_listing_parser,
    AirbnbListing,
    FinalListingData,
)
import requests
import os
from PIL import Image

OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY")
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
    inputs = processor(image, field, return_tensors="pt")
    outputs = catpion_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


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
        template="Are these two object describing the same listing? \n{format_instructions}\n{query}\n",
        input_variables=["query"],
    )
    model_name = "chatgpt-3.5"
    temperature = 0.0
    model = OpenAI(model_name=model_name, temperature=temperature)

    return model(_input.to_string())


def captioning_machine(images):
    captions = []
    for image in images:  # for each image in a listing
        caption = prediction_pipeline(image)
        captions.append(caption)
    return airbnb_prediction(captions)


# Streamlit app
def disambiguate(description, caption):
    # do something to the data to "disambiguate it"
    # most likely just prompt it like this:
    return disambiguate_query(description.json() + caption.json())


def create_final_text(description, captions, disambiguation, url, price, images, reviews):
    data = FinalListingData(
        description=description, caption=captions, matching_text_images=disambiguation, url=url, price=price, images=images, reviews=reviews
    )
    return create_document(data)


def process_listing(listing):
    raw_images = listing.get("photos")
    images = []
    for image in raw_images:
        im = Image.open(requests.get(image, stream=True).raw)
        images.append(im)

    description = listing.get(
        "description"
    )
    captions = captioning_machine(images)
    disambiguated = disambiguate(description, captions)
    return create_final_text(description, captions, disambiguated, listing.get("url"), listing.get("price"), raw_images[0], listing.get('reviews'))

def read_data():
    return dumps('documents.json')


def create_document(final_text):
    document =  Document(
        page_content=final_text.description,
        metadata={
            "captions": final_text.caption,
            "disambiguated": final_text.matching_text_images,
            "url": final_text.url,
            "price": final_text.price,
            "image_url": final_text.images,
            "reviews": final_text.reviews
        }
    )
    return document

def main():
    listings = read_data()
    p_listings = [process_listing(listing) for listing in listings]

    vectorstore = Chroma.from_documents(
        documents=p_listings, embedding=OpenAIEmbeddings()
    )
    vectorstore.save("AirbnbListings")
