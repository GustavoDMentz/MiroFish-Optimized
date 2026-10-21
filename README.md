<div align="center">

<img src="./static/image/MiroFish_logo_compressed.jpeg" alt="MiroFish Logo" width="70%"/>

# MiroFish — Optimized

**Versão aprimorada do [MiroFish](https://github.com/666ghj/MiroFish) com busca vetorial local, embeddings Google Gemini e armazenamento colunar em DuckDB.**

[![Based on](https://img.shields.io/badge/based%20on-666ghj%2FMiroFish-blue?style=flat-square&logo=github)](https://github.com/666ghj/MiroFish)
[![License](https://img.shields.io/github/license/666ghj/MiroFish?style=flat-square)](./LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Node](https://img.shields.io/badge/node-18+-339933?style=flat-square&logo=node.js&logoColor=white)](https://nodejs.org)

[English](#english) | [Português](#português)

</div>

---

## Português

### Segurança no uso local

Por padrão, o backend desativa o debug do Flask, escuta em `127.0.0.1` ao ser iniciado diretamente e permite CORS apenas para origens locais configuradas em `CORS_ORIGINS`. Corpos JSON das requisições não são registrados nos logs. A chave de sessão Flask é aleatória a cada processo quando `SECRET_KEY` não está definida; para manter sessões entre reinícios, configure uma chave forte e privada no `.env`.

O `docker-compose.yml` vincula as portas ao loopback do **host** (`127.0.0.1`). Isso não equivale a autenticação: não exponha o serviço diretamente à internet. **Atenção:** o Compose atual utiliza a imagem externa `ghcr.io/666ghj/mirofish:latest`; mudanças no código Python deste fork não alteram essa imagem. Para aplicar o endurecimento do backend, execute o código deste repositório ou construa uma imagem própria. A imagem externa também pode exigir configurações próprias.


> **Aviso de atribuição:** Este repositório é um fork aprimorado do projeto original **[666ghj/MiroFish](https://github.com/666ghj/MiroFish)**, desenvolvido pela equipe Shanda Group. Todo o crédito pela arquitetura e ideia original pertence aos autores do projeto base. As modificações aqui presentes visam expandir sua capacidade e acessibilidade.

### ⚡ O que é o MiroFish?

**MiroFish** é um motor de simulação e predição baseado em múltiplos agentes de IA. A partir de materiais-semente (notícias, relatórios, histórias), ele constrói um mundo digital paralelo de alta fidelidade onde milhares de agentes com memória, personalidade e lógica comportamental interagem livremente — permitindo que você simule e preveja eventos futuros como em um sandbox.

> Você fornece: materiais-semente e um objetivo de predição.  
> O MiroFish retorna: um relatório detalhado e um mundo digital interativo.

---

### 🚀 O que foi melhorado neste fork?

Este fork foca em tornar o MiroFish **mais robusto, rápido e barato de operar**, especialmente para usuários sem infraestrutura de banco de dados dedicada (Memgraph, Zep).

#### 1. 🔍 Busca Vetorial Local (DuckDB + NumPy)
O sistema original dependia exclusivamente do **Memgraph** para busca por similaridade vetorial. Este fork adiciona uma camada de busca vetorial puramente local usando **DuckDB** e **NumPy**, com similaridade de cosseno:

```
Memgraph vector search (se disponível)
    ↓ fallback
Busca vetorial local DuckDB + NumPy (NOVA)
    ↓ fallback
Busca FTS/LIKE no DuckDB
```

- Embeddings são armazenados nos atributos dos nós e arestas do grafo
- Funciona **sem Memgraph** instalado — ideal para desenvolvimento local
- Sem custo adicional de infraestrutura

#### 2. 🧠 Suporte a Google Gemini Embeddings

Configuração dedicada para embeddings separada do LLM principal, com suporte nativo à **API Google AI (Gemini)**:

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `EMBEDDING_API_KEY` | Chave da API de embeddings | Chave do Google AI Studio |
| `EMBEDDING_BASE_URL` | Endpoint da API | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `EMBEDDING_MODEL_NAME` | Modelo de embedding | `text-embedding-004` |

Isso permite usar **Gemini Embedding 2** (`text-embedding-004`) — um dos melhores embeddings disponíveis gratuitamente — para o grafo de conhecimento, enquanto usa outro modelo (ex.: Gemma, DeepSeek) para o LLM de simulação.

#### 3. 💾 Armazenamento Colunar com DuckDB
Substituição do armazenamento JSON por tabelas colunares **DuckDB**:
- Queries indexadas e INSERT/UPDATE eficientes
- Full-text search nativo
- Migração automática de dados JSON existentes na primeira carga
- Compatível com armazenamento em arquivo local (sem servidor)

#### 4. 🤖 Dois LLMs: Simulação + Aceleração

O sistema suporta dois modelos LLM configuráveis:

| Papel | Variável | Uso |
|-------|----------|-----|
| **LLM Principal** | `LLM_*` | Simulação de agentes, geração de relatório |
| **LLM de Aceleração** | `LLM_BOOST_*` | Tarefas rápidas (extração, classificação) |

Isso permite usar um modelo mais lento e capaz para raciocínio complexo, e um modelo rápido e barato para tarefas de apoio — reduzindo custo e latência.

#### 5. 🛡️ Tratamento de Falhas Robusto
- Erros de embedding são logados como `WARNING` e não interrompem a operação
- Todos os clientes externos (Memgraph, Zep, EmbeddingClient) têm fallback gracioso
- Logs detalhados com `mirofish.*` logger hierárquico

---

### 🏗️ Arquitetura

```
Frontend (Next.js / Node 18+)
    ↕ REST API
Backend (Flask / Python ≥3.11)
    ├── LLM Client (OpenAI-compatible: Gemini, DeepSeek, Qwen...)
    ├── Embedding Client (Google Gemini / qualquer OpenAI-compat.)
    ├── Local Graph Store
    │   ├── DuckDB (armazenamento colunar + FTS)
    │   ├── Busca vetorial local (NumPy cosine similarity)  ← NOVO
    │   └── Memgraph mirroring (opcional)
    ├── Zep Memory (opcional)
    └── OASIS Simulation Engine
```

---

### 🛠️ Instalação

#### Pré-requisitos

| Ferramenta | Versão | Verificação |
|------------|--------|-------------|
| Node.js | 18+ | `node -v` |
| Python | ≥3.11, ≤3.12 | `python --version` |
| uv | Mais recente | `uv --version` |

#### 1. Clone o repositório

```bash
git clone https://github.com/GustavoDMentz/MiroFish-Optimized.git
cd MiroFish-Optimized
```

#### 2. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` com suas chaves:

```env
# ===== LLM Principal (simulação de agentes) =====
# Compatível com qualquer API no formato OpenAI: Gemini, DeepSeek, Qwen, etc.
LLM_API_KEY=sua_chave_aqui
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL_NAME=gemma-3-27b-it

# ===== Embeddings (grafo de conhecimento) =====
# Recomendado: Google Gemini Embedding 2 (gratuito e de alta qualidade)
EMBEDDING_API_KEY=sua_chave_google_ai_studio
EMBEDDING_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
EMBEDDING_MODEL_NAME=text-embedding-004

# ===== LLM de Aceleração (opcional — tarefas rápidas) =====
LLM_BOOST_API_KEY=sua_chave
LLM_BOOST_BASE_URL=sua_url
LLM_BOOST_MODEL_NAME=gemma-3-4b-it

# ===== Memória Zep (opcional) =====
# Gratuito para uso básico: https://app.getzep.com/
ZEP_API_KEY=sua_chave_zep

# ===== Memgraph (opcional — busca vetorial em grafo) =====
# Deixe vazio para usar a busca vetorial local (DuckDB + NumPy)
MEMGRAPH_HOST=
MEMGRAPH_PORT=7687
```

> **Dica:** Para começar sem custo, use o [Google AI Studio](https://aistudio.google.com/) para obter chaves gratuitas para Gemini e Gemma.

#### 3. Instale as dependências

```bash
# Instala tudo de uma vez (Node + Python)
npm run setup:all
```

#### 4. Inicie os serviços

```bash
npm run dev
```

- **Frontend:** `http://localhost:3000`
- **Backend API:** `http://localhost:5001`

#### Docker (alternativa)

```bash
cp .env.example .env
# (edite o .env)
docker compose up -d
```

---

### 📋 Comparação com o projeto original

| Funcionalidade | Original (666ghj/MiroFish) | Este fork |
|----------------|---------------------------|-----------|
| Armazenamento de grafo | JSON | DuckDB colunar |
| Busca vetorial | Apenas Memgraph | Memgraph **+ local DuckDB/NumPy** |
| Embeddings | Configuração única LLM | API de embeddings separada |
| Suporte Google Gemini | Não nativo | ✅ Configuração dedicada |
| Fallback sem Memgraph | ❌ | ✅ Busca local |
| Dois LLMs (principal + boost) | ✅ | ✅ |
| Tratamento de erros de embedding | Básico | Robusto (WARNING + continuação) |

---

### 🤝 Contribuições

Contribuições são bem-vindas! Para melhorias que também se aplicam ao projeto original, considere abrir um PR em [666ghj/MiroFish](https://github.com/666ghj/MiroFish).

---

## English

> **Attribution notice:** This repository is an enhanced fork of **[666ghj/MiroFish](https://github.com/666ghj/MiroFish)**, originally developed by the Shanda Group team. All credit for the original architecture and concept belongs to the upstream authors. The modifications here aim to expand its capabilities and accessibility.

### ⚡ What is MiroFish?

**MiroFish** is a multi-agent AI simulation and prediction engine. From seed materials (news, reports, stories), it builds a high-fidelity parallel digital world where thousands of agents with memory, personality, and behavioral logic interact freely — letting you simulate and predict future events in a digital sandbox.

### 🚀 What's improved in this fork?

#### 1. 🔍 Local Vector Search (DuckDB + NumPy)
The original system relied exclusively on **Memgraph** for vector similarity search. This fork adds a purely local vector search layer using **DuckDB** and **NumPy** cosine similarity — no Memgraph required.

#### 2. 🧠 Google Gemini Embeddings Support
Dedicated embedding configuration, separate from the main LLM. Natively supports **Google AI API (Gemini)**, enabling use of **Gemini Embedding 2** (`text-embedding-004`) — one of the best freely available embedding models.

#### 3. 💾 Columnar Storage with DuckDB
JSON storage replaced with **DuckDB** columnar tables: indexed queries, efficient INSERT/UPDATE, native full-text search, automatic JSON migration on first load.

#### 4. 🛡️ Robust Error Handling
Embedding errors are logged as `WARNING` and don't interrupt operation. All external clients (Memgraph, Zep, EmbeddingClient) have graceful fallback.

### 🛠️ Quick Start

```bash
git clone https://github.com/GustavoDMentz/MiroFish-Optimized.git
cd MiroFish-Optimized
cp .env.example .env
# Edit .env with your API keys
npm run setup:all
npm run dev
```

- **Frontend:** `http://localhost:3000`
- **Backend API:** `http://localhost:5001`

---

## 📄 License & Attribution

This project is licensed under the same license as the original ([LICENSE](./LICENSE)).

Original project: **[666ghj/MiroFish](https://github.com/666ghj/MiroFish)**  
Original authors: [666ghj](https://github.com/666ghj) and the Shanda Group team  
Simulation engine: **[OASIS](https://github.com/camel-ai/oasis)** by CAMEL-AI  

Modifications in this fork by: [GustavoDMentz](https://github.com/GustavoDMentz)