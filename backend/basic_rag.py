from langchain.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores.utils import filter_complex_metadata  # Add this import

def load_documents(file_path):
    print(file_path)
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(file_path, mode="elements")
    else:
        raise ValueError("Unsupported file format")
    return loader.load_and_split()

pages = load_documents("3rdFeb.pptx")

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
texts = text_splitter.split_documents(pages)

# Filter out complex metadata values
filtered_texts = filter_complex_metadata(texts)  # Add this line

# 3. Create ChromaDB vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Create/persist ChromaDB using filtered documents
vector_db = Chroma.from_documents(
    documents=filtered_texts,
    embedding=embeddings,
    persist_directory="chroma_db"
)
vector_db.persist()

# 4. RAG Query Function
def rag_query(query: str, k: int = 10):
    # Reload ChromaDB (for subsequent runs)
    vector_db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    
    # Retrieve relevant chunks
    docs = vector_db.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-14B",  # Or any other DeepSeek model (distilled)
        max_seq_length=3000,
        dtype=None, 
        load_in_4bit=True,  # 4-bit quantization for inference
        token = os.getenv("HUGGINGFACE_API_TOKEN"),
    )
    
    FastLanguageModel.for_inference(model)
    
    # Format prompt
    prompt = f"""Answer the question using only this context by first explaining your reasoning step-by-step within <think> tags, then provide the final answer in <answer> tags:
    
    {context}
    
    Question: {query}"""
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=3000-1100,
        temperature=0.7,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5. Example Usage
answer = rag_query("Do you agree to include the following into the 11bn SFD?")
print("---------------------------------------------------------------------------------------------------------------------------------")
print("Response:")
print(answer)