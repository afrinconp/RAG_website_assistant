# RAG Bank Assistant with Web Scraping

This project is a technical assessment solution for a Machine Learning Engineer / AI Engineer role. It implements a complete RAG system that scrapes public banking website content, stores raw and cleaned data locally, indexes the content in a vector database, exposes a minimal conversational UI, persists conversation history by session ID, and includes analytics over historical conversations.

The default target website is BBVA Colombia: `https://www.bbva.com.co/`. The system can also be configured to scrape another bank website by changing environment variables.

## Main Features

- Web scraping with `requests` and `BeautifulSoup`.
- Raw and processed local storage.
- Text cleaning and chunking pipeline.
- Embeddings using a free multilingual SentenceTransformer model.
- Local vector database using ChromaDB.
- Explicit semantic retrieval that finds the chunks most similar to the user prompt.
- Optional reranker using a CrossEncoder model.
- Gemini answer generation through LangChain.
- Streamlit conversational interface.
- Persistent conversation memory in SQLite.
- Configurable number of previous messages used as context.
- Analytics dashboard over historical conversations.
- Dockerized execution with one command.
- At least 3 documented design patterns.

## Architecture

```text
Website
  ↓
ScraperFactory -> BankScraper
  ↓
data/raw/pages.jsonl
  ↓
Cleaner + Chunker
  ↓
data/processed/chunks.jsonl
  ↓
EmbeddingModel + ChromaVectorStore
  ↓
RetrieverStrategy + Optional Reranker
  ↓
ConversationService + SQLite Memory
  ↓
Streamlit UI + Analytics Dashboard
```

## Stack and Justification

| Component | Technology | Reason |
|---|---|---|
| Language | Python | Required by the assessment and standard for ML/RAG systems. |
| UI | Streamlit | Minimal, fast, clean conversational interface. |
| Scraping | requests + BeautifulSoup | Simple and reliable for static public pages. |
| Embeddings | SentenceTransformers | Free, local, multilingual-friendly. |
| Vector DB | ChromaDB | Self-hosted, simple, local persistence. |
| Memory | SQLite | Lightweight local persistence for conversations. |
| Analytics | pandas + Plotly | Fast exploration of historical conversations. |
| Docker | Dockerfile + docker-compose | Reproducible execution with one command. |
| LLM orchestration | LangChain | Clean RAG prompt orchestration and provider abstraction. |
| Generator | Gemini | Generates grounded answers from retrieved chunks. |

## Design Patterns Used

### 1. Factory Pattern

Applied in `app/scraper/factory.py`.

It creates the correct scraper implementation based on configuration:

```python
scraper = ScraperFactory.create(bank_name=settings.bank_name)
```

Why: this allows the system to support BBVA today and another bank tomorrow without changing the rest of the pipeline.

### 2. Strategy Pattern

Applied in `app/rag/retrieval.py`.

The retrieval logic is abstracted behind a strategy interface. The current implementation supports semantic retrieval and optional reranking.

Why: retrieval behavior can evolve independently, for example by adding hybrid search, BM25, or metadata filtering.

### 3. Singleton Pattern

Applied in `app/memory/database.py` and `app/rag/vector_store.py`.

The SQLite and Chroma clients are reused instead of being recreated multiple times.

Why: this avoids unnecessary resource initialization and centralizes access to persistent services.

## Requirements

- Docker
- Docker Compose
- Internet access during the first run to download embedding/reranker models and to call Gemini

Optional for local development without Docker:

- Python 3.11+
- virtualenv or conda

## Setup with Docker

Clone the repository:

```bash
git clone https://github.com/afrinconp/RAG_website_assistant.git
cd rag_bank_assistant
```


Build and run:

```bash
docker compose up --build
```

Open the app:

```text
http://localhost:8501
```

## Recommended First Run

Inside the Streamlit UI, use the sidebar buttons in this order:

1. Run scraper
2. Process scraped data
3. Build vector index
4. Ask questions in the chat
5. Open the analytics tab

## Running Locally without Docker

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
cp .env.example .env
streamlit run app/ui/streamlit_app.py
```

## Configuration

Configuration is controlled through `.env`:

```env
BANK_NAME=bbva
START_URL=https://www.bbva.com.co/
MAX_PAGES=20
N_HISTORY_MESSAGES=6
CHUNK_SIZE=900
CHUNK_OVERLAP=150
TOP_K=8
FINAL_K=4
USE_RERANKER=true
```

## Conversation Memory

Each conversation uses a `session_id`. The system stores:

- session ID
- role: user or assistant
- message content
- timestamp
- retrieved sources when available

The number of previous messages injected into the prompt is controlled by:

```env
N_HISTORY_MESSAGES=6
```

## Analytics

The analytics tab reads the SQLite conversation history and computes:

- total messages
- number of sessions
- messages by role
- messages per day
- average message length
- most common words in user questions

These metrics can help estimate impact, adoption, user needs, and possible improvements to the knowledge base.

## Known Limitations

- The default answer generator is extractive and template-based to avoid requiring a paid LLM API.
- If the selected website heavily depends on JavaScript, `requests` may not capture all content. Selenium or Playwright could be added later.
- The scraper intentionally limits the number of pages to avoid overloading public websites.
- Local embedding/reranker model download can take time during first execution.

## Future Improvements

- Add an optional Gemini/OpenAI/Ollama LLM provider.
- Add Playwright for JavaScript-heavy pages.
- Add hybrid retrieval with BM25 + embeddings.
- Add automated evaluation for RAG quality.
- Add authentication and user-level session management.
- Add CI tests and linting.

## Suggested Commit History

For the real submission, avoid a single large commit. A good commit progression would be:

```text
feat: initialize project structure and docker setup
feat: add configurable web scraper
feat: persist raw and processed scraped data
feat: add chunking and vector indexing pipeline
feat: implement retrieval and reranking strategy
feat: add conversation memory with sqlite
feat: add streamlit chat interface
feat: add conversation analytics dashboard
docs: document design patterns and setup instructions
```

## Hugging Face generation model

The chat now uses a real open-source Hugging Face generation model as the final RAG generation step.

Default model:

```env
HF_GENERATION_MODEL=google/flan-t5-small
MAX_NEW_TOKENS=220
```

RAG flow:

```text
User question -> embedding -> vector search -> optional reranker -> retrieved context -> Hugging Face generator -> final answer
```

The Streamlit UI shows two things:

1. **Respuesta final** generated by the Hugging Face model.
2. **Fuentes recuperadas** with the chunks used as context.

You can replace the model in `.env` with another open-source text2text model from Hugging Face. For example, a larger FLAN-T5 model may improve answer quality but will require more RAM and slower inference.




## RAG Retrieval and Gemini Generation

The code that retrieves the chunks most similar to the prompt is located in:

```text
app/rag/vector_store.py
app/rag/retrieval.py
```

The key method is:

```python
ChromaVectorStore.search(query, k)
```

It performs this flow:

```text
user prompt
→ SentenceTransformer embedding
→ Chroma vector search
→ top-k most similar chunks
→ optional CrossEncoder reranking
→ final-k chunks
→ Gemini via LangChain
```

The LangChain + Gemini generation code is located in:

```text
app/rag/generator.py
```

The RAG generation flow is:

```text
question
→ retrieved_context
→ conversation_history
→ ChatPromptTemplate
→ ChatGoogleGenerativeAI
→ grounded final answer
```

Required environment variable:

```bash
GOOGLE_API_KEY=your_google_api_key
```

Default Gemini model:

```bash
GEMINI_MODEL=gemini-3.1-flash-lite
```

You can change it in `.env` if needed.


### Option B: enter the Gemini API key in Streamlit

The Streamlit interface also includes a sidebar field where the user can paste the Gemini API key during the session.

Steps:

1. Open `http://localhost:8501`
2. Go to the sidebar section **Gemini API Key**
3. Paste the key
4. Click **Save Gemini API key**
5. Ask questions in the chat

This is useful for demos because the evaluator does not need to edit the `.env` file manually.


The Streamlit sidebar includes a direct link to create or copy a Gemini API key:

```text
https://aistudio.google.com/api-keys
```


## Reranker safety note

The core RAG retrieval uses Chroma vector similarity and does not depend on the reranker.

`USE_RERANKER=false` is the default because some local Docker environments can fail when loading
`CrossEncoder` models with Torch/Transformers meta-tensor errors.

If you want to test reranking, set:

```bash
USE_RERANKER=true
```

If the reranker fails, the app now falls back automatically to vector similarity retrieval instead
of crashing.
