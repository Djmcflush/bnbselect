from segmentmodel import caption_model as CaptionModel
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
caption_model, processor = CaptionModel()
read_index = os.environ.get("READ_INDEX")
write_index = os.environ.get("WRITE_INDEX")


def prediction_pipeline(image, iterate_over_airbnb_fields=False):
    captions = []
    if not iterate_over_airbnb_fields:
        captions =  caption_prediction(image, field="Describe All the objects in this image")
        return captions
    # captions = []
    # for field in AirbnbListing.__fields__:
    #     caption = caption_prediction(image, field.description)
    #     captions.append(caption)
    #return " ".join(captions)


def caption_prediction(image, field):
    inputs = processor(image, field, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


def airbnb_prediction(query):
    parser = airbnb_output_parser()
    #if type(query) == List:
    final_query = " ".join(query)
    
    # prompt = PromptTemplate(
    #     template="Describe all the features and amenities inside of this description.\n{format_instructions}\n{query}\n",
    #     input_variables=["query"],
    #     partial_variables={"format_instructions": parser.get_format_instructions()},
    # )
    temperature = 0.0
    model_name = "text-davinci-003"
    model = OpenAI(model_name=model_name, temperature=temperature)

    # _input = prompt.format_prompt(query=query)
    result = model(final_query)
    #result = parser.parse(output)
    return result


def disambiguate_query(query):
    if type(query) == List:
        query = " ".join(query)
    query_prompt= f"Describe all the features and amenities inside of this description.\n{query}\n"
    # prompt = PromptTemplate(
    #     template="Are these two object describing the same listing? \n{format_instructions}\n{query}\n",
    #     input_variables=["query"],
    # )
    # _input = prompt.format_prompt(query=query)

    model_name = "text-davinci-003"
    temperature = 0.0
    model = OpenAI(model_name=model_name, temperature=temperature)

    return model(query_prompt)


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
    return disambiguate_query(description + caption)


def create_final_text(description, captions, disambiguated_text, url, price, images, reviews):
    data = FinalListingData(
        description=description, caption=captions, disambiguated_text=disambiguated_text, url=url, price=price, images=images, reviews=reviews
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
    with open('documents.json', 'r') as f:
        documents = json.load(f)
    return documents
def create_document(final_text):
    document =  Document(
        page_content=final_text.description,
        metadata={
            "captions": final_text.caption,
            "disambiguated": final_text.disambiguated_text,
            "url": final_text.url,
            "price": final_text.price,
            "image_url": final_text.images,
            "reviews": final_text.reviews
        }
    )

    return document
import json
def write_to_file(listing):
    with open("processed_listings.json", "a") as f:
        json.dump(listing, f)
        f.write('\n')

def main():
    listings = read_data()
    for listing in listings:
        p_listing = process_listing(listing)
        write_to_file(p_listing.json())


if __name__ == "__main__":
    main()


    