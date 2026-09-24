import os
os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")
os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/hf_cache/hub")
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

file_path = "./Attention_Is_All_You_Need.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)


all_splits = text_splitter.split_documents(docs)


embedding_model = HuggingFaceEmbeddings(
  model_name = "sentence-transformers/all-mpnet-base-v2"
)

vector_store = Chroma(
  collection_name="example_collection",      
  embedding_function=embedding_model,            
  persist_directory="./chroma_langchain_db" 
)
document_ids = vector_store.add_documents(documents=all_splits)

sample = vector_store.get(
    limit=1,
    include=["embeddings", "documents", "metadatas"]
)

def retrieve_context(query):
  retrieved_docs = vector_store.similarity_search(query)

  docs_content = ""

  for doc in retrieved_docs:
    docs_content += f"Source: {doc.metadata}\n"
    docs_content += f"Content: {doc.page_content}\n\n"

  return docs_content, retrieved_docs

model = init_chat_model(
  "google_genai:gemini-3.6-flash",
  api_key=GOOGLE_API_KEY,
)


def ask_about_pdf(user_query):

  doc_content, retrieved_docs = retrieve_context(user_query)

  system_message = f"""
  You are a helpful chatbot.

  Use only the following pieces of context to answer the question.
  Don't make up any new information.

  Context:
  {doc_content}
  """

  messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_query}
  ]

  response = model.invoke(messages)

  return {
        "answer":response.content,
        "source_documents": retrieved_docs,
        "context_used": doc_content
  }



    

result = ask_about_pdf(
    "What improvements have been made to attention mechanisms since 2017?"
)
print("Answer/n")
print(result["answer"][0]["text"])