from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import requests
import os
from datetime import datetime, timedelta
from scrubber import scrub_text

load_dotenv()

# ---- State ----
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    needs_fresh_data: bool
    messages: List

# ---- Tools ----
def search_database(state: AgentState) -> AgentState:
    print("  [Agent] Searching local database...")
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = Chroma(persist_directory="./data/chroma", embedding_function=embeddings)
    docs = vectorstore.similarity_search(state["question"], k=5)
    context = "\n".join([doc.page_content for doc in docs])
    return {**state, "context": context}

def fetch_fresh_data(state: AgentState) -> AgentState:
    print("  [Agent] Fetching fresh data from SAM.gov...")
    url = "https://api.sam.gov/opportunities/v2/search"
    all_fresh = []
    offset = 0
    limit = 1000

    today = datetime.today()
    one_year_ago = today - timedelta(days=364)

    while True:
        params = {
            "api_key": os.getenv("SAM_API_KEY"),
            "limit": limit,
            "offset": offset,
            "postedFrom": one_year_ago.strftime("%m/%d/%Y"),
            "postedTo": today.strftime("%m/%d/%Y"),
        }
        response = requests.get(url, params=params)

        if response.status_code == 429:
            print("  [Agent] API rate limited — using local data only")
            break

        if response.status_code != 200:
            print(f"  [Agent] API error {response.status_code} — stopping")
            break

        data = response.json()
        opportunities = data.get("opportunitiesData", [])

        if not opportunities:
            break

        for item in opportunities:
            title = item.get('title', '')
            if "multiple award schedule" in title.lower():
                continue
            all_fresh.append(f"Title: {title}\nAgency: {item.get('fullParentPathName', '')}\n")

        print(f"  [Agent] Fetched {len(all_fresh)} fresh records so far...")

        if len(opportunities) < limit:
            break

        offset += limit

    fresh_context = "\n".join(all_fresh)
    combined = state["context"] + "\n" + fresh_context
    return {**state, "context": combined}

def generate_answer(state: AgentState) -> AgentState:
    print("  [Agent] Generating answer...")
    llm = OllamaLLM(model="mistral")
    prompt = f"""
    You are a government contracting assistant. Use the following context from SAM.gov 
    to answer the question. Only use provided context, do not make anything up.
    If you don't know, say so.

    Context: {state["context"]}
    Question: {state["question"]}
    Answer:
    """
    answer = llm.invoke(prompt)
    return {**state, "answer": answer}

def decide_needs_fresh_data(state: AgentState) -> AgentState:
    print("  [Agent] Deciding whether to fetch fresh data...")

    # Scrub the question before processing
    scrubbed_question, detected = scrub_text(state["question"])
    if detected:
        print(f"  [Agent] PII detected and scrubbed: {detected}")

    llm = OllamaLLM(model="mistral")
    prompt = f"""
    A user asked: "{scrubbed_question}"
    The local database returned this context: "{state["context"][:500]}"
    
    Does the context seem relevant and sufficient to answer the question?
    Reply with only YES or NO.
    """
    result = llm.invoke(prompt).strip().upper()
    needs_fresh = "NO" in result
    print(f"  [Agent] Context sufficient: {'No' if needs_fresh else 'Yes'}")
    return {**state, "question": scrubbed_question, "needs_fresh_data": needs_fresh}

# ---- Routing ----
def route(state: AgentState) -> str:
    if state["needs_fresh_data"]:
        return "fetch_fresh"
    return "answer"

# ---- Build Graph ----
def build_agent():
    graph = StateGraph(AgentState)
    
    graph.add_node("search_db", search_database)
    graph.add_node("decide", decide_needs_fresh_data)
    graph.add_node("fetch_fresh", fetch_fresh_data)
    graph.add_node("answer", generate_answer)
    
    graph.set_entry_point("search_db")
    graph.add_edge("search_db", "decide")
    graph.add_conditional_edges("decide", route, {
        "fetch_fresh": "fetch_fresh",
        "answer": "answer"
    })
    graph.add_edge("fetch_fresh", "answer")
    graph.add_edge("answer", END)
    
    return graph.compile()

if __name__ == "__main__":
    agent = build_agent()
    print("SAM.gov Agent ready! Type 'quit' to exit\n")
    
    while True:
        question = input("Ask a question: ")
        if question.lower() == "quit":
            break
        print("\nThinking...\n")
        result = agent.invoke({
            "question": question,
            "context": "",
            "answer": "",
            "needs_fresh_data": False,
            "messages": []
        })
        print(f"\nAnswer: {result['answer']}\n")