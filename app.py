import os
import shutil
import streamlit as st
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# BASIC SETUP

load_dotenv()
st.set_page_config(page_title="RAG Chatbot Pro", layout="wide")
st.title("🧠 Intelligent RAG Chatbot")

UPLOAD_DIR = "./temp_docs"

# COMPACT SIDEBAR LOGIC (No Scrolling Fix)

with st.sidebar:
    st.title("🤖 Config & Upload")
    
    # SETTINGS (COLLAPSIBLE) ---
   
    with st.expander("⚙️ Settings", expanded=False):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            api_key = st.text_input("Groq API Key:", type="password")

        # Model Selector
        selected_model = st.selectbox(
            "🤖 AI Model:",
            ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
            index=0
        )
        
        # Clear Chat Button 
        if st.button("🗑️ Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()

    # UPLOAD 
    st.subheader("📂 Upload Files")
    uploaded_files = st.file_uploader(
        "Drop PDFs/TXT/DOCX here:", 
        type=['pdf', 'txt', 'docx'], 
        accept_multiple_files=True,
        label_visibility="collapsed" 
    )

    #  ACTION BUTTONS 
    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("🔄 Process", use_container_width=True)
    with col2:
        summarize_btn = st.button("📜 Summary", use_container_width=True)

    # Processing Logic
    if process_btn or summarize_btn:
        if not api_key:
            st.error("❌ API Key Required inside Settings!")
        elif uploaded_files:
            with st.spinner("Processing..."):
                if os.path.exists(UPLOAD_DIR):
                    shutil.rmtree(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR)

                raw_documents = []
                for file in uploaded_files:
                    file_path = os.path.join(UPLOAD_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    if file.name.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                    elif file.name.endswith(".docx"):
                        loader = Docx2txtLoader(file_path)
                    elif file.name.endswith(".txt"):
                        loader = TextLoader(file_path)
                    
                    raw_documents.extend(loader.load())

                if not raw_documents:
                    st.error("⚠️ Empty File!")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    final_docs = text_splitter.split_documents(raw_documents)
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
                    st.success("✅ Done!")
                    
                    if summarize_btn:
                        llm_sum = ChatGroq(groq_api_key=api_key, model_name=selected_model)
                        sum_prompt = ChatPromptTemplate.from_template("Summarize in 5 bullet points:\n\n{context}")
                        full_text = " ".join([d.page_content for d in raw_documents])[:4000]
                        chain = sum_prompt | llm_sum | StrOutputParser()
                        res = chain.invoke({"context": full_text})
                        st.session_state.messages.append({"role": "assistant", "content": f"**Summary:**\n{res}"})
        else:
            st.warning("⚠️ Select files first.")


# MAIN CHAT LOGIC

if not api_key:
    st.info("👈 Please set your API Key in the sidebar Settings.")
    st.stop()

# Initialize LLM with Selected Model
llm = ChatGroq(groq_api_key=api_key, model_name=selected_model)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask something...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if "vectors" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please process documents first.")
    else:
        #  RAG PIPELINE 
        retriever = st.session_state.vectors.as_retriever()
        
        context_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        context_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", context_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Keep the answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        qa_chain = qa_prompt | llm | StrOutputParser()

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                try:
                    retrieved_docs = history_aware_retriever.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    
                    stream = qa_chain.stream({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history,
                        "context": formatted_context
                    })
                    
                    for chunk in stream:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)

                    with st.expander("📚 Source Citations"):
                        for i, doc in enumerate(retrieved_docs):
                            page_num = doc.metadata.get('page', 'N/A')
                            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                            st.write(f"**Source {i+1}:** {source} (Page {page_num})")
                            st.caption(doc.page_content[:150] + "...")

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.chat_history.extend(
                        [HumanMessage(content=user_query), AIMessage(content=full_response)]
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
