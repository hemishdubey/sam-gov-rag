# SAM.gov Local LLM Assistant

A fully local AI assistant for querying U.S. government contract opportunities from SAM.gov — built with Mistral, LangChain, LangGraph, and ChromaDB.

## The Problem: OpenClaw and Data Leakage
Tools like OpenClaw send queries and contract data to external cloud APIs, creating serious privacy and security risks for organizations working with proprietary or sensitive information. For government contractors, this is unacceptable — RFPs, bid strategies, and internal research cannot leave the organization.

## The Solution
This tool runs 100% locally. No data ever leaves your machine:
- Queries are processed by a local Mistral 7B model via Ollama
- Contract data is stored in a local ChromaDB vector database
- The LangGraph agent makes decisions entirely on-device
- A PII scrubber automatically strips sensitive data before it ever reaches the LLM
- Works fully offline once data is ingested

## Features
- 🔒 **Fully local** — no cloud APIs, no data leakage
- 🤖 **LangGraph agent** — intelligently decides whether to use local database or fetch fresh data from SAM.gov
- 🛡️ **PII scrubber** — automatically detects and removes names, emails, phone numbers, SSNs, credit cards, and more before processing
- 🔄 **Automated ingestion** — scheduler rebuilds the database daily with fresh contracts
- 💬 **Chat UI** — clean Streamlit interface with routing transparency and live PII warnings
- 🗄️ **Graceful failover** — database backup and restore if ingestion fails

## Tech Stack
| Component | Technology |
|---|---|
| LLM | Mistral 7B via Ollama |
| Agent framework | LangGraph |
| RAG pipeline | LangChain |
| Vector database | ChromaDB |
| PII scrubbing | Microsoft Presidio + Regex |
| UI | Streamlit |
| Data source | SAM.gov public API |

## Why This vs OpenClaw
| Feature | OpenClaw | This Tool |
|---|---|---|
| Data leaves machine | ✅ Yes | ❌ Never |
| Requires cloud API to query | ✅ Yes | ❌ No |
| PII protection | ❌ No | ✅ Yes |
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
git clone https://github.com/hemishdubey/sam-gov-rag
cd sam-gov-rag
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
ollama pull mistral
python -m spacy download en_core_web_lg
```

### Configuration
Create a `.env` file:

### Run
Ingest data first:
```bash
python ingest.py
```

Start the scheduler in one terminal:
```bash
python scheduler.py
```

Start the app in another terminal:
```bash
streamlit run app.py
```

## How the PII Scrubber Works
Every query is scrubbed before it reaches the LLM:
1. Regex patterns catch SSNs and credit card numbers
2. Microsoft Presidio detects names, emails, phone numbers, IPs, passport numbers, and more
3. Detected entities are replaced with typed placeholders e.g. `<PERSON>`, `<EMAIL_ADDRESS>`
4. The UI shows a warning if sensitive data was detected and removed