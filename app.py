import streamlit as st
from dotenv import load_dotenv
from agent import build_agent

load_dotenv()

st.set_page_config(page_title="SAM.gov AI Assistant", page_icon="🏛️", layout="centered")

st.title("🏛️ SAM.gov Contract Assistant")
st.caption("Powered by Mistral — fully local, no data leaves your machine")

# Privacy badge
st.success("🔒 100% Local — No data ever leaves your machine")

@st.cache_resource
def load_agent():
    return build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "route" in message:
            st.caption(message["route"])

if prompt := st.chat_input("Ask about government contracts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            agent = load_agent()
            result = agent.invoke({
                "question": prompt,
                "context": "",
                "answer": "",
                "needs_fresh_data": False,
                "messages": []
            })
            response = result["answer"]
            route_used = "🌐 Fetched fresh data from SAM.gov" if result["needs_fresh_data"] else "💾 Answered from local database"
            st.markdown(response)
            st.caption(route_used)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "route": route_used
    })