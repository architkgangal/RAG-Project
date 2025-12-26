from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms import OpenAI
import chromadb
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from IPython.display import Markdown
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
import tqdm as notebook_tqdm
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


## Define Embedding Function, LLM and Other Parameters
# Many ways of getting embeddings: https://docs.llamaindex.ai/en/latest/examples/embeddings/ollama_embedding/
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cuda:0")
# Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

## Define LLM
Settings.llm = Ollama(model="llama3", request_timeout=120.0)

## Define Splitter
Settings.node_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=512)

## Define Other Settings
# number of tokens reserved for text generation.
Settings.num_output = 2048
# maximum input size to the LLM
Settings.context_window = 8000



## Load up the Vvector DB
# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db") # Persistant

# Create collection
chroma_collection = db.get_or_create_collection("DB_collection")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Change default storage context to new vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./chroma_db/index_save") # <-- Give the persit dir used to save the index

# Load the vector store index from stored vectors
index = VectorStoreIndex.from_documents([], # <-- Change this to [] instead of documents if loading an already created and stored index
                                        storage_context=storage_context)




## Define the Query Engine
# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(
    # response_mode="tree_summarize",
    response_mode="compact",
    # response_mode="refine",
)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)




## Start Making Queries
response = query_engine.query("Get me studies that have looked at knowledge management in hospitals and/or healthcare")
display(Markdown(f"{response}"))