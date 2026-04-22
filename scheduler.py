import schedule
import time
import shutil
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

API_KEY = os.getenv("SAM_API_KEY")

def fetch_sam_opportunities():
    print("  [Scheduler] Fetching all data from SAM.gov...")

    url = "https://api.sam.gov/opportunities/v2/search"
    all_opportunities = []
    offset = 0
    limit = 1000
    total_fetched = 0

    today = datetime.today()
    one_year_ago = today - timedelta(days=364)

    while True:
        params = {
            "api_key": API_KEY,
            "limit": limit,
            "offset": offset,
            "postedFrom": one_year_ago.strftime("%m/%d/%Y"),
            "postedTo": today.strftime("%m/%d/%Y"),
        }

        response = requests.get(url, params=params)

        if response.status_code == 429:
            print("  [Scheduler] API rate limited — will retry next cycle")
            return []

        if response.status_code != 200:
            print(f"  [Scheduler] API error {response.status_code} — stopping pagination")
            break

        data = response.json()
        opportunities_data = data.get("opportunitiesData", [])

        if not opportunities_data:
            print("  [Scheduler] No more data available")
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

        total_fetched += len(opportunities_data)
        print(f"  [Scheduler] Fetched {total_fetched} records so far (offset {offset})...")

        # If we got less than the limit we've hit the end
        if len(opportunities_data) < limit:
            print("  [Scheduler] Reached end of available data")
            break

        offset += limit

    print(f"  [Scheduler] Total fetched: {len(all_opportunities)} opportunities")
    return all_opportunities

def rebuild_database():
    print("\n[Scheduler] Starting scheduled ingestion...")

    if os.path.exists("./data/chroma"):
        shutil.copytree("./data/chroma", "./data/chroma_backup", dirs_exist_ok=True)
        print("  [Scheduler] Backed up old database")

    texts = fetch_sam_opportunities()
    if not texts:
        print("  [Scheduler] No data fetched — keeping existing database, will retry next cycle")
        return

    try:
        shutil.rmtree("./data/chroma")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.create_documents(texts)
        embeddings = OllamaEmbeddings(model="mistral")
        Chroma.from_documents(chunks, embeddings, persist_directory="./data/chroma")
        print(f"  [Scheduler] Stored {len(chunks)} chunks in ChromaDB")
        print("  [Scheduler] Database rebuilt successfully!")
        shutil.rmtree("./data/chroma_backup", ignore_errors=True)
    except Exception as e:
        print(f"  [Scheduler] Error rebuilding database: {e}")
        print("  [Scheduler] Restoring backup...")
        shutil.copytree("./data/chroma_backup", "./data/chroma", dirs_exist_ok=True)

schedule.every().day.at("00:00").do(rebuild_database)

print("[Scheduler] Starting up — running initial ingestion...")
rebuild_database()

print("\n[Scheduler] Running! Database will refresh daily at midnight.")
print("Keep this running in a separate terminal alongside app.py\n")

while True:
    schedule.run_pending()
    time.sleep(60)