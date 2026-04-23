# Python Docs RAG Chatbot

Chatbot conversacional que responde perguntas sobre a [documentação oficial do Python (pt-br)](https://docs.python.org/pt-br/3/) usando RAG (Retrieval-Augmented Generation).

## Como funciona

1. **Crawling**: no primeiro uso, um crawler BFS rastreia toda a documentação do Python a partir de `docs.python.org/pt-br/3/`, extrai parágrafos e itens de lista de cada página
2. **Embeddings**: os chunks de texto são convertidos em vetores semânticos com `SentenceTransformer (all-MiniLM-L6-v2)` e salvos em cache no disco
3. **Retrieval**: a pergunta do usuário é vetorizada e comparada por similaridade de cosseno contra todos os chunks; os 5 mais relevantes são selecionados
4. **Generation**: o contexto recuperado + histórico da conversa são enviados ao modelo `llama-3.3-70b-versatile` via API Groq, que gera a resposta final

## Pré-requisitos

- Python 3.10+
- Chave de API Groq gratuita: [console.groq.com](https://console.groq.com)

## Instalação

```bash
# Criar e ativar ambiente virtual
python -m venv .venv
.venv\Scripts\activate

# Instalar dependências
pip install sentence-transformers numpy groq fastapi uvicorn pydantic httpx beautifulsoup4 python-dotenv
```

## Configuração

Edite o arquivo `.env` com sua chave Groq:

```
GROQ_API_KEY=gsk_sua-chave-aqui
```

## Uso

**Opção 1 — Launcher automático (recomendado no Windows):**
```
start.bat
```
Inicia o servidor e abre o browser automaticamente em `http://localhost:8000`.

**Opção 2 — Manual:**
```bash
.venv\Scripts\uvicorn llm:app --reload
```
Depois abra `http://localhost:8000` no browser.

> **Primeira execução:** o crawling completo da documentação leva 10–20 minutos. As próximas inicializações carregam o cache em segundos.

## Estrutura

| Arquivo | Descrição |
|---|---|
| `llm.py` | Servidor principal: crawler BFS, indexação, endpoints `/chat` e `/reset` |
| `index.html` | Interface do chatbot (HTML/JS puro, sem build) |
| `fastrag.py` | Versão simplificada com 4 URLs fixas — referência didática do padrão RAG |
| `launch.py` | Launcher que abre o browser somente quando o servidor estiver pronto |
| `start.bat` | Atalho Windows para `launch.py` |
| `.env` | Chave de API (não versionado) |
| `.cache/` | Chunks e embeddings gerados automaticamente (não versionado) |

## API

```
POST /chat
Body:     { "message": "O que são list comprehensions?" }
Response: { "answer": "...", "sources": ["[título] trecho...", ...] }

POST /reset
Response: { "status": "ok" }
```
