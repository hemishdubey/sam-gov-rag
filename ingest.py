import requests
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

API_KEY = os.getenv("SAM_API_KEY")

def fetch_sam_opportunities():
    url = "https://api.sam.gov/opportunities/v2/search"
    params = {
        "api_key": API_KEY,
        "limit": 1000,
        "postedFrom": "01/01/2024",
        "postedTo": "12/31/2024",
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    keywords = ["cyber", "security", "information technology", "software", "IT", "network", "cloud", "data"]
    
    opportunities = []
    for item in data.get("opportunitiesData", []):
        title = item.get('title', '').lower()
        description = item.get('description', '').lower()
        
        if any(kw.lower() in title or kw.lower() in description for kw in keywords):
            if "multiple award schedule" in title.lower():
                continue
            print(f"  - {item.get('title', '')}")
            text = f"""
            Title: {item.get('title', '')}
            Agency: {item.get('fullParentPathName', '')}
            Type: {item.get('type', '')}
            Description: {item.get('description', '')}
            Posted Date: {item.get('postedDate', '')}
            """
            opportunities.append(text)
    
    return opportunities

def store_in_chromadb(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(texts)
    
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./data/chroma")
    print(f"Stored {len(chunks)} chunks in ChromaDB")

if __name__ == "__main__":
    print("Fetching SAM.gov opportunities...")
    texts = fetch_sam_opportunities()
    print(f"Fetched {len(texts)} opportunities")
    store_in_chromadb(texts)
    print("Done!")