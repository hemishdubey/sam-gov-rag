# SAM.gov Local LLM Assistant

A fully local AI assistant for querying U.S. government contract opportunities from SAM.gov — built with Mistral, LangChain, LangGraph, and ChromaDB.

## The Problem: OpenClaw and Data Leakage
Tools like OpenClaw send queries and contract data to external cloud APIs, creating serious privacy and security risks for organizations working with proprietary or sensitive information. For government contractors, this is unacceptable — RFPs, bid strategies, and internal research cannot leave the organization.

## The Solution
This tool runs 100% locally. No data ever leaves your machine:
- Queries are processed by a local Mistral 7B model via Ollama
- Contract data is stored in a local ChromaDB vector database
- The LangGraph agent makes decisions entirely on-device
- Works fully offline once data is ingested

## Features
- 🔒 **Fully local** — no cloud APIs, no data leakage
- 🤖 **LangGraph agent** — intelligently decides whether to use local database or fetch fresh data from SAM.gov
- 🔄 **Automated ingestion** — scheduler rebuilds the database daily with fresh contracts
- 💬 **Chat UI** — clean Streamlit interface with routing transparency
- 🛡️ **Graceful failover** — database backup and restore if ingestion fails

## Tech Stack
| Component | Technology |
|---|---|
| LLM | Mistral 7B via Ollama |
| Agent framework | LangGraph |
| RAG pipeline | LangChain |
| Vector database | ChromaDB |
| UI | Streamlit |
| Data source | SAM.gov public API |

## Why This vs OpenClaw
| Feature | OpenClaw | This Tool |
|---|---|---|
| Data leaves machine | ✅ Yes | ❌ Never |
| Requires cloud API to query | ✅ Yes | ❌ No |
| Works offline | ❌ No | ✅ Yes |
| Self-hostable | ❌ No | ✅ Yes |
| Automatic data refresh | ❌ No | ✅ Yes |

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed
- SAM.gov API key (free at [sam.gov](https://sam.gov))

### Installation
```bash
git clone https://github.com/yourusername/sam-gov-rag
cd sam-gov-rag
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
ollama pull mistral
```

### Configuration
Create a `.env` file:


### Run
Start the scheduler in one terminal:
```bash
python scheduler.py
```

Start the app in another terminal:
```bash
streamlit run app.py
```

## Project Structure