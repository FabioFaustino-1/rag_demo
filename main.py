from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Carregar PDF
loader = PyPDFLoader(
    "documentos/livro.pdf"
)

documentos = loader.load()

print(f"Quantidade de páginas: {len(documentos)}")


# 2. Dividir em chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documentos)

print(f"Quantidade de chunks: {len(chunks)}")


# 3. Criar modelo de embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Criar banco vetorial
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Chunks armazenados no ChromaDB!")