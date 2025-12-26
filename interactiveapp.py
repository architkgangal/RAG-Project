import streamlit as st
import time
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb

# Display the UI
st.title("MIS Journal RAG Bot")

# Dropdown to select models
model_options = ["llama3", "mistral", "gpt4"]
selected_model = st.selectbox("Select a model", model_options)

# Initialize session states
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions from users.
    Your tone should be professional and informative.
    Context: {context}
    History: {history}
    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://localhost:11434", model=selected_model, verbose=True,
                                  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    print(f"Model selected: {selected_model}")

# Initialize ChromaDB connection and retriever using LangChain's Chroma integration
if 'vectorstore' not in st.session_state:
    # Use Hugging Face embeddings
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Initialize ChromaDB connection
    db = chromadb.PersistentClient(path="C:/Users/Tejin/Downloads/chroma_db/chroma_db")

    try:
        # Attempt to get or create the collection
        chroma_collection = db.get_or_create_collection("DB_collection")
    except chromadb.errors.InvalidCollectionException:
        # If collection doesn't exist, create a new one
        chroma_collection = db.get_or_create_collection("DB_collection")

    # Set up the vector store retriever with the new collection
    st.session_state.vectorstore = Chroma(
        collection_name="DB_collection",
        embedding_function=embedding_function,
        persist_directory="C:/Users/Tejin/Downloads/chroma_db/chroma_db"
    )
    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

# Initialize chat history if not present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        chain_type='stuff',
        retriever=st.session_state.retriever,  # Use the Chroma retriever
        verbose=True,
        chain_type_kwargs={"verbose": True, "prompt": st.session_state.prompt, "memory": st.session_state.memory}
    )


# Chat history display
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# User input for chat
user_input = st.chat_input("You:", key="user_input")
if user_input:
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            # Query the QA chain using the user's input
            response = st.session_state.qa_chain(user_input)

        message_placeholder = st.empty()
        full_response = ""
        for chunk in response['result'].split():
            full_response += chunk + " "
            if len(full_response) % 5 == 0:
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
        message_placeholder.markdown(full_response)

    chatbot_message = {"role": "assistant", "message": response['result']}
    st.session_state.chat_history.append(chatbot_message)
else:
    st.write("No input provided. Please ask a question.")
