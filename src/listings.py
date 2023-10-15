import pinecone

import os

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_AIRBNB_RAW_INDEX = os.environ.get("PINECONE_AIRBNB_RAW_INDEX")


def read_listings_from_pinecone_db():
    # Assuming pinecone is already imported and initialized
    return pinecone.fetch(PINECONE_AIRBNB_RAW_INDEX)


def fetch_top_k_from_pinecone_db(index_name, query_vector, top_k):
    return pinecone.query(index_name=index_name, queries=query_vector, top_k=top_k)


def embed_text_with_pinecone(text):
    # Assuming pinecone is already imported and initialized
    return pinecone.compute_text_embeddings(text)
