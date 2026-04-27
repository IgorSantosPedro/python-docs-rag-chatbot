from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urljoin
import asyncio
import json
import numpy as np
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_URL   = "https://docs.python.org/pt-br/3/"
CACHE_DIR       = Path(__file__).parent / ".cache"
CHUNKS_FILE     = CACHE_DIR / "chunks.json"
EMBEDDINGS_FILE = CACHE_DIR / "embeddings.npy"

# Páginas de navegação pura — não contêm conteúdo útil para o RAG
SKIP_PATTERNS = ("genindex", "modindex", "search", "404")

SYSTEM_PROMPT = (
    "Você é um assistente especializado na documentação do Python. "
    "Responda de forma clara e didática, sempre em português. "
    "Use apenas as informações fornecidas no contexto de cada mensagem. "
    "Se a resposta não estiver no contexto, diga que não encontrou essa informação "
    "na documentação indexada."
)

emb_model: SentenceTransformer = None
doc_embeddings: np.ndarray = None
documents: list[str] = []
history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]


# ── Crawler BFS ────────────────────────────────────────────────────────────────

async def crawl_all_docs() -> list[str]:
    """
    Rastreia toda a documentação via BFS a partir de BASE_URL.
    Em cada página: extrai novos links internos E os chunks de texto — tudo em uma passagem.
    """
    visited: set[str] = {BASE_URL}
    queue: list[str]  = [BASE_URL]
    all_chunks: list[str] = []
    processed = 0

    # Limita a 10 requisições simultâneas para não sobrecarregar o servidor
    semaphore = asyncio.Semaphore(10)

    async def process_page(client: httpx.AsyncClient, url: str) -> tuple[list[str], list[str]]:
        """Retorna (novos_links, chunks) para uma página."""
        async with semaphore:
            try:
                resp = await client.get(url, timeout=20)
            except Exception:
                return [], []

        soup = BeautifulSoup(resp.text, "html.parser")

        # Descobre novos links internos para continuar o BFS
        new_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0]          # remove fragmento de âncora
            if not href.endswith(".html"):
                continue
            abs_url = urljoin(str(resp.url), href)
            if abs_url.startswith(BASE_URL) and not any(p in abs_url for p in SKIP_PATTERNS):
                new_links.append(abs_url)

        # Extrai chunks de texto do conteúdo principal da página
        title_tag = soup.find("h1")
        title   = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]
        content = soup.find("div", {"class": "body"}) or soup.find("main") or soup.body

        chunks = []
        if content:
            for tag in content.find_all(["p", "li"]):
                text = tag.get_text(separator=" ", strip=True)
                if len(text) >= 80:
                    chunks.append(f"[{title}] {text}")

        return new_links, chunks

    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Processa o BFS em lotes: cada lote é o conjunto atual da fila
        while queue:
            batch  = queue[:]
            queue.clear()

            results = await asyncio.gather(*[process_page(client, u) for u in batch])

            for url, (new_links, chunks) in zip(batch, results):
                processed += 1
                print(f"  [{processed:>4}] {url.replace(BASE_URL, '')} → {len(chunks)} chunks")
                all_chunks.extend(chunks)

                for link in new_links:
                    if link not in visited:
                        visited.add(link)
                        queue.append(link)

    return all_chunks


# ── Startup ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global emb_model, doc_embeddings, documents

    # Passo 1 — Carrega o modelo de embeddings
    print("Carregando modelo de embeddings...")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    if CHUNKS_FILE.exists() and EMBEDDINGS_FILE.exists():
        # Passo 2a — Cache existe: carrega do disco (segundos)
        print("Cache encontrado, carregando índice do disco...")
        documents.extend(json.loads(CHUNKS_FILE.read_text(encoding="utf-8")))
        doc_embeddings = np.load(EMBEDDINGS_FILE)
        print(f"  {len(documents)} chunks carregados.\n")
    else:
        # Passo 2b — Sem cache: rastreia toda a documentação (somente na primeira vez)
        print(f"\nPrimeiro uso — rastreando {BASE_URL}")
        print("Isso pode levar 10–20 minutos. Não feche o terminal.\n")

        documents.extend(await crawl_all_docs())
        print(f"\nTotal: {len(documents)} chunks. Calculando embeddings...")

        # Passo 3 — Computa embeddings e salva cache no disco
        doc_embeddings = emb_model.encode(
            documents, normalize_embeddings=True, show_progress_bar=True
        )
        CACHE_DIR.mkdir(exist_ok=True)
        CHUNKS_FILE.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
        np.save(EMBEDDINGS_FILE, doc_embeddings)
        print("Cache salvo em .cache/ — próximas inicializações serão instantâneas.\n")

    print("Pronto! Servidor disponível em http://localhost:8000\n")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Recuperação ────────────────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = 5) -> list[str]:
    query_vec = emb_model.encode([question], normalize_embeddings=True)
    scores    = (doc_embeddings @ query_vec.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in top_indices]


# ── Endpoints ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/page")
def page():
    return FileResponse(Path(__file__).parent / "page.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    sources = retrieve(request.message)
    context = "\n".join(f"- {chunk}" for chunk in sources)

    user_message = f"Contexto da documentação:\n{context}\n\nPergunta: {request.message}"
    history.append({"role": "user", "content": user_message})

    try:
        client = Groq()
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=512,
            messages=history,
        )
    except Exception as e:
        history.pop()
        raise HTTPException(status_code=502, detail=str(e))

    answer = completion.choices[0].message.content
    history.append({"role": "assistant", "content": answer})

    return ChatResponse(answer=answer, sources=sources)


@app.post("/reset")
def reset():
    history.clear()
    history.append({"role": "system", "content": SYSTEM_PROMPT})
    return {"status": "ok"}
