import streamlit as st
from langchain.vectorstores import Chroma
from PIL import Image
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import json
import os
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings()

def main():
    st.title("Retrieval Access Generation Pipeline")

    st.sidebar.title("Options")
    st.sidebar.subheader("Record Display Options")
    num_listings = st.sidebar.slider("Number of Records to display", 1, 6, 4)

    st.header("Describe the record you need")
    query = st.text_input("")
    decoder = json.JSONDecoder()
    json_query = []
    with open('processed_listings.json', 'r') as f:
        for jsonObj in f:
                json_ = json.loads(jsonObj)
                json_query.append(json_)
    docs = json_query


    documents = []
    for doc in docs:

        doc = json.loads(doc)
        print("THIS IS PAGE CONTENT ", doc.get('page_content'))
        print("THIS IS METADATA ", doc.get('metadata'))
        print(type(doc))
        print(doc.keys())
        reviews = doc.get('metadata').get('reviews')
        reviews = " ".join(reviews)
        metadata = doc.get('metadata')
        metadata['reviews'] = reviews
        try:
            document =  Document(
            page_content=doc.get('page_content'),
            metadata=metadata,
        )
            documents.append(document)
        except:
            pass
        print(len(documents))
    # llm = HuggingFacePipeline(pipeline=pipeline)


    chroma_search = Chroma.from_documents(
        documents=documents, embedding=embeddings
    )    # chroma_search = Chroma.load("AirbnbListings")
    chroma_search.as_retriever(
        search_type="mmr", search_kwargs={"k": num_listings, "lambda_mult": 0.25}
    )
    resutls = []

    if query:
        results = chroma_search.similarity_search(query)
        st.write("")
        st.write("Processing...")

        st.subheader("Generated Records")
        # st.write(results)
        images = []
        for result in results:
            raw_image = result.metadata.get("image_url")
            images.append(Image.open(requests.get(raw_image, stream=True).raw))

        st.write("Here are the records that match your description:")
        for i, result in enumerate(results):
            st.write(f"Listing {i+1}:")
            st.image(images[i])
            st.write(result.page_content)
            st.write(result.metadata.get("clasification", "CONFIDENTIAL"))
            st.write(result.metadata.get("NTK", "NTK11, NTK13"))
            st.write(result.metadata.get("url"))

        st.write("Records processed and deleted.")


if __name__ == "__main__":
    main()

