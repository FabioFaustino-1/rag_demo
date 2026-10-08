from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from google import genai

# =========================
# 1. Carregar o banco
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


# ===========================
# 1.5 Carregar dotenv
# ===========================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# =========================
# 2. Fazer a pergunta
# =========================

pergunta = "What is virtual memory?"


# =========================
# 3. Buscar no ChromaDB
# =========================

THRESHOLD = 0.8

resultados = db.similarity_search_with_score(
    pergunta,
    k=3
)

resultados_relevantes = [
    (documento, score)
    for documento, score in resultados
    if score <= THRESHOLD
]


# =========================
# 4. Verificar resultados
# =========================

if not resultados_relevantes:

    print("Não encontrei informações suficientemente relevantes.")

else:

    print(f"Encontrei {len(resultados_relevantes)} resultados relevantes.")

    # =========================
    # 5. Montar o contexto
    # =========================

    contexto = ""

    for i, (documento, score) in enumerate(resultados_relevantes, start=1):

        pagina = documento.metadata.get("page")

        contexto += f"""
--- Trecho {i} | Página {pagina} ---

{documento.page_content}

"""

    print("\n================ CONTEXTO ================\n")
    print(contexto)


    prompt = f"""
Você é um assistente que responde perguntas usando exclusivamente
as informações fornecidas no contexto.

Se a resposta não estiver no contexto, diga:
"Não encontrei informações suficientes no documento."

Contexto:

{contexto}

Pergunta:

{pergunta}
"""

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)

print("\n================ RESPOSTA ================\n")
print(response.text)