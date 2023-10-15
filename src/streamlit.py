import streamlit as st
from langchain.vectorstores import Chroma
from PIL import Image


def main():
    st.title("Retrieval Access Generation Pipeline")

    st.sidebar.title("Options")
    st.sidebar.subheader("Listings Display Options")
    num_listings = st.sidebar.slider("Number of listings to display", 1, 6, 4)

    st.header("Describe the Airbnb Listing you want")
    query = st.text_input("Describe your perfect item")

    chroma_search = Chroma.load("AirbnbListings")
    chroma_search.as_retriever(
        search_type="mmr", search_kwargs={"k": num_listings, "lambda_mult": 0.25}
    )
    resutls = []

    if query is not None:
        results = chroma_search.get_relevant_documents(query)
        st.write("")
        st.write("Processing...")

        st.subheader("Generated Captions")
        # st.write(results)
        images = []
        for result in results:
            raw_image = result.get("photos")[0]
            images.append(Image.open(requests.get(raw_image, stream=True).raw))

        st.write("Here are the listings that match your description:")
        for i, result in enumerate(results):
            st.write(f"Listing {i+1}:")
            st.image(images[i])
            st.write(result.get("description"))
            st.write(results.get("hyperlink"))

        st.write("Listings processed and saved.")


if __name__ == "__main__":
    main()
