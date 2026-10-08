# RAG Demo — ChromaDB + Gemini

Projeto demonstrativo de um sistema RAG (Retrieval-Augmented Generation).

O projeto utiliza:

- Python
- LangChain
- PyPDF
- Hugging Face Embeddings
- ChromaDB
- Google Gemini API

O fluxo do projeto é:

PDF → texto → chunks → embeddings → ChromaDB → busca semântica → contexto → Gemini → resposta

---

## 1. Requisitos

É necessário ter instalado:

- Python 3.10 ou superior
- Git

---

## 2. Clonar o projeto

Clone o repositório:

```bash
git clone https://github.com/FabioFaustino-1/rag_demo.git
```
Entrar na pasta

```bash
cd rag_demo
```
Criar ambiente virtual

```bash
python -m venv .env
```
Ative o ambiente virtual
```cmd
.venv\Scripts\activate
```
instale as dependencias
```bash
pip install -r requirements.txt
```
adicione um documento pdf a pasta \documentos

crie um arquivo .env
```env
GEMINI_API_KEY=SUA_CHAVE_AQUI
```
altere o main.py com o nome do arquivo que anteriormente foi adicionado aos documentos

execute o main.py

abra o app.py

altere a pergunta na variavel "pergunta" com a linguagem equivalente a do livro

após isto execute o app.py



























