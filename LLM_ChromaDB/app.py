import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# ---- Streamlit UI Setup ---- #
st.set_page_config(layout="wide")
st.title("Local Web QA Bot 🔍")
st.sidebar.header("Settings")
MODEL = st.sidebar.selectbox("Choose a Model", ["llama2-uncensored", "llama3", "deepseek"], index=0)
MAX_HISTORY = st.sidebar.number_input("Max Chat History", 1, 10, 3)
CONTEXT_SIZE = st.sidebar.number_input("Context Size", 1024, 8192, 4096, step=1024)

# ---- Session State ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state or st.session_state.get("prev_context_size") != CONTEXT_SIZE:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.prev_context_size = CONTEXT_SIZE

# ---- LLM and Embeddings ---- #
llm_model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# ---- Load Data or Reload Persisted Chroma ---- #
persist_directory = "./chroma_db"
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# ---- RetrievalQA Chain ---- #
qa = RetrievalQA.from_chain_type(llm=llm_model, chain_type="stuff", retriever=retriever, return_source_documents=True)

# ---- Display Chat History ---- #
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Trim Memory ---- #
def trim_memory():
    while len(st.session_state.chat_history) > MAX_HISTORY * 2:
        st.session_state.chat_history.pop(0)

# ---- Handle User Prompt ---- #
if prompt := st.chat_input("Ask a question about the websites..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    trim_memory()

    with st.chat_message("assistant"):
        response_container = st.empty()

        # Fetch answer from LLM
        try:
            result = qa({"query": prompt})
            answer = result.get("result", "No response generated.")
        except Exception as e:
            answer = f"Error: {str(e)}"

        response_container.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        trim_memory()
