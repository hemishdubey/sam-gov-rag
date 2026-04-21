import requests
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

API_KEY = os.getenv("SAM_API_KEY")

def fetch_sam_opportunities():
    print("Fetching all data from SAM.gov...")
    url = "https://api.sam.gov/opportunities/v2/search"
    all_opportunities = []
    offset = 0
    limit = 1000

    while True:
        params = {
            "api_key": API_KEY,
            "limit": limit,
            "offset": offset,
            "postedFrom": "01/01/2000",
            "postedTo": datetime.today().strftime("%m/%d/%Y"),
        }

        response = requests.get(url, params=params)

        if response.status_code == 429:
            print("API rate limited — try again tomorrow")
            return []

        if response.status_code != 200:
            print(f"API error {response.status_code} — stopping")
            break

        data = response.json()
        opportunities_data = data.get("opportunitiesData", [])

        if not opportunities_data:
            print("No more data available")
            break

        for item in opportunities_data:
            title = item.get('title', '')
            if "multiple award schedule" in title.lower():
                continue
            text = f"""
            Title: {title}
            Agency: {item.get('fullParentPathName', '')}
            Type: {item.get('type', '')}
            Description: {item.get('description', '')}
            Posted Date: {item.get('postedDate', '')}
            """
            all_opportunities.append(text)

        print(f"Fetched {len(all_opportunities)} records so far (offset {offset})...")

        if len(opportunities_data) < limit:
            print("Reached end of available data")
            break

        offset += limit

    return all_opportunities

def store_in_chromadb(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(texts)
    embeddings = OllamaEmbeddings(model="mistral")
    Chroma.from_documents(chunks, embeddings, persist_directory="./data/chroma")
    print(f"Stored {len(chunks)} chunks in ChromaDB")

if __name__ == "__main__":
    texts = fetch_sam_opportunities()
    if texts:
        store_in_chromadb(texts)
        print("Done!")
    else:
        print("No data fetched — database unchanged")