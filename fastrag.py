from contextlib import asynccontextmanager
import numpy as np
import httpx
from bs4 import BeautifulSoup
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# URLs das páginas da documentação oficial do Python (pt-br) que serão indexadas.
# Pode-se adicionar ou trocar qualquer página do mesmo domínio.
DOC_URLS = [
    "https://docs.python.org/pt-br/3/tutorial/introduction.html",
    "https://docs.python.org/pt-br/3/tutorial/datastructures.html",
    "https://docs.python.org/pt-br/3/library/functions.html",
    "https://docs.python.org/pt-br/3/tutorial/errors.html",
]

# Variáveis globais preenchidas no startup da aplicação
model: SentenceTransformer = None
doc_embeddings: np.ndarray = None
documents: list[str] = []


def fetch_and_chunk(url: str) -> list[str]:
    # Busca o HTML da página e extrai o título para usar como prefixo dos chunks
    response = httpx.get(url, follow_redirects=True, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # A documentação do Python usa a div "body" como área de conteúdo principal
    content = soup.find("div", {"class": "body"}) or soup.find("main") or soup.body

    chunks = []
    for tag in content.find_all(["p", "li"]):
        text = tag.get_text(separator=" ", strip=True)
        # Ignora chunks muito curtos (navegação, rótulos, etc.)
        if len(text) >= 80:
            # Prefixa o título da página para dar contexto ao chunk durante a busca
            chunks.append(f"[{title}] {text}" if title else text)

    return chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, doc_embeddings, documents

    # Passo 1 — Carrega o modelo de embeddings.
    # "all-MiniLM-L6-v2" é um modelo leve que transforma texto em vetores de 384 dimensões.
    # Na primeira execução ele é baixado automaticamente do HuggingFace (~90 MB).
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Passo 2 — Busca e processa as páginas da documentação do Python.
    # Cada URL é acessada, o HTML é parseado com BeautifulSoup e o conteúdo é
    # dividido em chunks por parágrafo/item de lista, prefixados com o título da página.
    print(f"Indexando {len(DOC_URLS)} página(s) da documentação...")
    for url in DOC_URLS:
        chunks = fetch_and_chunk(url)
        documents.extend(chunks)
        print(f"  {url} → {len(chunks)} chunks")

    print(f"Total: {len(documents)} chunks indexados.")

    # Passo 3 — Pré-computa os embeddings de todos os chunks extraídos.
    # Cada chunk vira um vetor numérico que representa seu significado semântico.
    # normalize_embeddings=True deixa todos os vetores com comprimento 1,
    # o que permite usar produto escalar como medida de similaridade (mais rápido).
    doc_embeddings = model.encode(
        documents, normalize_embeddings=True, show_progress_bar=True
    )

    yield  # A aplicação fica rodando aqui até ser encerrada


app = FastAPI(lifespan=lifespan)


def retrieve(question: str, top_k: int = 3) -> list[str]:
    # Passo 4 — Converte a pergunta do usuário em vetor usando o mesmo modelo.
    query_embedding = model.encode([question], normalize_embeddings=True)

    # Passo 5 — Calcula a similaridade entre a pergunta e cada chunk.
    # O produto escalar entre vetores normalizados equivale ao cosseno do ângulo entre eles:
    # quanto mais próximo de 1, mais semanticamente similar é o par (pergunta, chunk).
    scores = (doc_embeddings @ query_embedding.T).flatten()

    # Passo 6 — Seleciona os top_k chunks com maior pontuação de similaridade.
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in top_indices]


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]  # Chunks da documentação usados como contexto


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")

    # Passo 7 — Recuperação (Retrieval): busca os chunks mais relevantes para a pergunta.
    sources = retrieve(request.question)

    # Passo 8 — Monta o contexto: junta os chunks recuperados em um único bloco de texto
    # que será injetado no prompt enviado ao LLM.
    context = "\n".join(f"- {chunk}" for chunk in sources)

    # Passo 9 — Geração (Augmented Generation): envia o contexto + pergunta ao ChatGPT.
    # O modelo é instruído a responder APENAS com base no contexto fornecido,
    # evitando que ele "invente" informações além do que está nos documentos.
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Use apenas o contexto abaixo para responder à pergunta. "
                    f"Se a resposta não estiver no contexto, diga que não sabe.\n\n"
                    f"Contexto:\n{context}\n\n"
                    f"Pergunta: {request.question}"
                ),
            }
        ],
    )

    # Passo 10 — Retorna a resposta gerada pelo LLM junto com as fontes usadas.
    return QuestionResponse(
        answer=completion.choices[0].message.content,
        sources=sources,
    )
