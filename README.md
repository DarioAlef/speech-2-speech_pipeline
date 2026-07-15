# Voice Chat Local — Tutor de Inglês por Voz

Pipeline speech-2-speech com modelos locais e leves para treinar inglês.

## Objetivo

Ajudar um falante de português a **praticar inglês falado**. O assistente:

- Responde em **inglês simples** por padrão (com um botão para trocar para **português** quando você quiser entender melhor).
- **Corrige** seus erros de gramática e de escolha de palavra, mostrando a forma natural de dizer.
- **Corrige a pronúncia**: além de entender as palavras, ele analisa os sons que você produziu e aponta erros de pronúncia (ex.: o "th" de *think* dito como /t/).
- Mostra a **legenda** do que ele falou na tela.
- Nunca usa emojis (o texto é lido em voz alta).

## Como funciona (cascata modular)

```
Microfone → WebSocket → detecção de voz → transcrição (Whisper)
        → LLM (Qwen3-4B, professor) → divisão por frase
        → síntese de voz (Piper, voz EN/PT) → reprodução no navegador
```

A resposta é falada **frase por frase** enquanto o modelo ainda escreve, o que dá a sensação de conversa (latência de ~1,5–2,5 s).

## Requisitos

- Ubuntu 24.04 com driver NVIDIA + CUDA (testado em RTX 3050, 6 GB VRAM) e ~32 GB RAM
- Python 3.11+
- ~10 GB de disco livre (modelos)
- Navegador moderno (Chrome/Edge/Firefox) com permissão de microfone

---

## Setup inicial (uma vez)

> Tudo é instalado dentro da pasta `.venv` do projeto — **nada global**.

```bash
# 1. Ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Dependências
pip install -r requirements.txt

# 3. llama-cpp-python com CUDA (o pacote padrão é só CPU)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# 4. Baixar os modelos para dentro do projeto (LLM, vozes, VAD, pronúncia)
python scripts/download_models.py
```

O Whisper (transcrição) baixa sozinho na primeira vez que você fala.

---

## Como subir o serviço (depois de tudo pronto)

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Abra **http://localhost:8000**, permita o microfone, **segure o botão** para falar e solte para ouvir a resposta.

> Para acessar de outro aparelho na rede, o navegador exige HTTPS. Rode com certificado:
> `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem`

---

## Configuração

Tudo o que muda o comportamento fica em [`config.yaml`](./config.yaml) (sem mexer no código): idioma, voz, tamanho/temperatura do modelo, prompt do professor, análise de pronúncia, porta do servidor. Dá para sobrescrever com um arquivo `.env` (veja `.env.example`).

## Testes

```bash
.venv/bin/pytest -q
```

## Dicas rápidas

- **Erro na inicialização citando um arquivo**: rode `python scripts/download_models.py`.
- **CUDA out of memory**: baixe `llm.n_ctx` (ex.: 2048) no `config.yaml`.
- **Respostas muito lentas**: o `llama-cpp-python` provavelmente ficou só em CPU — refaça o passo 3 do setup.
- **Sem áudio no navegador**: a reprodução precisa de um clique; o primeiro toque no botão já resolve.
