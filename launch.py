"""
Launcher do Python Docs Chatbot.
Inicia o uvicorn em background e só abre o browser quando o servidor estiver pronto.
"""
import subprocess
import sys
import time
import webbrowser
import urllib.request
import urllib.error
from pathlib import Path

ROOT    = Path(__file__).parent
PYTHON  = ROOT / ".venv" / "Scripts" / "python.exe"
UVICORN = ROOT / ".venv" / "Scripts" / "uvicorn.exe"
URL     = "http://localhost:8000"
PAGE    = "http://localhost:8000/page"

print("=" * 50)
print("  Python Docs Chatbot")
print("=" * 50)
print("\nIniciando servidor...")
print("(Na primeira vez pode demorar 1-2 minutos para")
print(" baixar o modelo e indexar a documentação)\n")

# Inicia o uvicorn mostrando o output dele neste mesmo terminal
proc = subprocess.Popen(
    [str(UVICORN), "llm:app"],
    cwd=ROOT,
)

# Fica tentando conectar até o servidor responder
print("Aguardando o servidor ficar pronto", end="", flush=True)
while True:
    try:
        urllib.request.urlopen(URL, timeout=2)
        break  # servidor respondeu
    except (urllib.error.URLError, OSError):
        # ainda não está pronto — tenta de novo em 2 segundos
        print(".", end="", flush=True)
        time.sleep(2)
    except Exception:
        # qualquer outro erro (ex: resposta HTTP de erro) significa que o servidor subiu
        break

print("\n\nServidor pronto! Abrindo browser...\n")
webbrowser.open(PAGE)

# Mantém o terminal aberto e exibe o log do servidor
try:
    proc.wait()
except KeyboardInterrupt:
    print("\nEncerrando servidor...")
    proc.terminate()
    sys.exit(0)
