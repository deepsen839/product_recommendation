# Occasion-Based Product Recommendation System

A complete take-home implementation of an occasion-aware product discovery application. It accepts a free-text occasion, ranks products from the supplied catalog, and displays image-led product cards in the same page. It includes a Streamlit UI, FastAPI backend, hybrid recommendation engine, tests, and Docker setup.

## 1. Features

- Free-text occasion input plus quick-select examples (Birthday, Wedding, Anniversary, Diwali, Valentine's Day, Office Party, Housewarming, Farewell).
- Hybrid ranking: Hugging Face Sentence Transformers semantic embeddings, TF-IDF cosine similarity, deterministic occasion vocabulary expansion and lexical overlap.
- Ranked results with relevance score, match explanation, image, product name, inferred category, brand, description, price, discount, SKU and stock.
- Empty-input validation and no-results state with suggestions to broaden the query.
- FastAPI endpoints for health, catalog browsing, product detail and recommendations.
- Eligibility filtering so draft, review-required, unapproved, deleted or out-of-stock listings are not recommended.
- Docker Compose starts UI and API independently.

## 2. Catalog mapping and data-quality decisions

The supplied `products.json` is an array of 110 records. It uses `id`, `name`, `shortDescription`, `mainImage`, `brand`, `categoryId`, `price`, `discountPercentage`, `stock`, `isApproved`, `status` and other commerce fields. The assignment PDF describes a different/abstract schema (`product_id`, `title`, `files`, `vendor_category_desc`, etc.). The implementation maps the actual JSON keys and does not assume the PDF sample keys exist.

At build time the catalog contains **110 total products**, of which **104 are eligible** (ACTIVE, approved, not deleted, and stock greater than zero). The remaining six listings are excluded because they are `REVIEW_REQUIRED` or `DRAFT` and/or not approved. This is a deliberate marketplace-safe default; change `_eligible()` in `app/recommender.py` if the business owner wants a different publication policy.

The JSON has `categoryId` but no category name/description lookup table. Therefore the UI derives a human-readable category from product name/description using transparent keyword rules. It does not pretend that these inferred labels are the original vendor category descriptions. Add a category lookup table when one is available.

## 3. Recommendation approach

1. Normalize the user occasion and expand known occasions using `OCCASION_TERMS` (for example Diwali expands to Deepavali, festival, festive, dry fruits, hamper and marigold).
2. Build a product document from name, short description, description, brand and slug.
3. Fit a TF-IDF vectorizer with unigram/bigram features and English stop-word removal.
4. Encode eligible product documents with Hugging Face `sentence-transformers/all-MiniLM-L6-v2` on CPU; encode the expanded occasion at query time.
5. Compute cosine similarity from normalized sentence embeddings, alongside TF-IDF cosine similarity and lexical overlap.
6. When embeddings are available, combine scores as `0.60 * embedding_similarity + 0.25 * tfidf_similarity + 0.15 * lexical_overlap`. If the model cannot load, fall back to `0.78 * tfidf_similarity + 0.22 * lexical_overlap`.
7. Sort descending, attach an explanation, and return the requested number of results.

Sentence Transformers provides semantic matching for related phrases that may not share exact words. Groq remains an optional query-understanding layer; it is not the embedding model. Embeddings are generated locally on CPU and the model is downloaded from Hugging Face the first time recommendations are requested. The implementation does not require a vector database for this 110-product catalog.

## 4. Project structure

```text
occasion_recommender/
├── app/
│   ├── __init__.py
│   ├── api.py                 # FastAPI routes
│   └── recommender.py         # catalog loader, eligibility, ranking
├── data/
│   └── products.json          # supplied source catalog
├── tests/
│   └── test_recommender.py
├── frontend/                  # React + Vite frontend (primary UI)
│   ├── src/main.jsx
│   ├── src/styles.css
│   ├── Dockerfile
│   └── nginx.conf
├── streamlit_app.py           # Optional Streamlit UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 5. Run locally

Python 3.10 or 3.11 is recommended.

```bash
# from the project root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# UI
streamlit run streamlit_app.py

# API (in another terminal)
uvicorn app.api:app --reload
```

Open the UI at `http://localhost:8501`. Open interactive API docs at `http://localhost:8000/docs`.

## 6. Run with Docker

```bash
docker compose up --build
```

- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- API health: `http://localhost:8000/health`

Stop with `docker compose down`.

## 7. API usage

### Health

```bash
curl http://localhost:8000/health
```

### Recommendations

```bash
curl -X POST http://localhost:8000/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"occasion":"Diwali","limit":8}'
```

Response shape:

```json
{
  "occasion": "Diwali",
  "count": "number of returned products",
  "recommendations": [
    {
      "id": "...",
      "name": "Diwali Dry fruits hampers",
      "category": "Gift Hampers & Packaging",
      "relevance_score": 30.0,
      "match_reason": "..."
    }
  ]
}
```

The actual count and scores depend on the catalog and query. The response includes the original product fields plus `category`, `relevance_score` and `match_reason`.

### Browse eligible products

```bash
curl 'http://localhost:8000/products?limit=20'
```

### Product detail

```bash
curl 'http://localhost:8000/products/PRODUCT_ID'
```

Returns HTTP 404 if the ID is not present.

## 8. Tests

```bash
pytest -q
```

Tests cover catalog loading/filtering, relevant Diwali matching, blank input, and result limits.

## 9. Evaluation mapping

| Assignment requirement | Implementation |
|---|---|
| Occasion input | Streamlit text input and quick occasion selector |
| Compute relevance | Hybrid TF-IDF + lexical score |
| Ranked recommendations | Descending relevance score |
| Product title/category/description/image | Product cards |
| Visible search action | Find gifts submit button |
| Results below input | Same-page responsive grid |
| Empty state | Input validation and no-match message |
| Clean code | Separated engine, API, UI and tests |
| README | Setup, design, API and test instructions |
| Bonus semantic approach | Hugging Face Sentence Transformers embeddings combined with TF-IDF and keyword matching |

## 10. Screenshots and submission checklist

The assignment requests two real UI screenshots and a public/shared Git repository. Capture them after launching the UI:

1. **Occasion input screenshot:** show the page header, occasion input and Find gifts button before/while entering an occasion.
2. **Product results screenshot:** enter `Diwali`, click Find gifts, and capture the ranked product cards below the input.

Save the screenshots as `screenshots/occasion-input.png` and `screenshots/product-results.png` before submission. These are intentionally not fabricated in this source bundle; capture them from the running application so they show the actual UI and loaded catalog.

Before submitting:
- [ ] Run `pytest -q`.
- [ ] Run the UI and API locally.
- [ ] Capture the two screenshots above.
- [ ] Create a GitHub repository and push this project (do not upload the confidential assignment PDF unless the hiring team explicitly requests it).
- [ ] Share the repository link and screenshots through the hiring team's requested channel.

## 11. Hugging Face Sentence Transformers setup

The recommendation engine uses `sentence-transformers/all-MiniLM-L6-v2` for semantic embeddings by default. It runs on CPU. The first recommendation request downloads the model from Hugging Face and caches it in the container/user cache. This requires outbound internet access on first use. To disable transformer embeddings and use the lightweight local fallback, set:

```env
ENABLE_SENTENCE_TRANSFORMERS=false
```

You can choose another compatible Sentence Transformers model with `SENTENCE_TRANSFORMER_MODEL`. The API health endpoint reports whether embeddings are enabled and whether the model has loaded. If the optional library/model is unavailable, the service logs a warning and continues with TF-IDF plus keyword matching.

## 12. Libraries and AI-assistant disclosure

- FastAPI: API framework.
- Streamlit: interactive UI.
- scikit-learn: TF-IDF vectorization and cosine similarity.
- Hugging Face Sentence Transformers: semantic text embeddings (`all-MiniLM-L6-v2`).
- Groq: optional LLM query expansion (not used for embeddings).
- NumPy/SciPy: numerical dependencies.
- pytest: automated tests.
- Docker: reproducible packaging.

AI coding assistant: ChatGPT was used to assist with implementation and documentation; the implementation should be reviewed, tested and understood by the submitting candidate before submission.


## React frontend

The primary frontend is a responsive React + Vite application in `frontend/`.
It provides occasion search, popular occasion shortcuts, responsive product cards,
product images, brand/category/description, price and discount, relevance scores,
and expandable recommendation explanations. It calls the FastAPI backend through
the same-origin `/api` path; Nginx proxies those requests to the `api` container.

### Run React locally (without Docker)

Start the API in one terminal:

```bash
pip install -r requirements.txt
uvicorn app.api:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite development server proxies `/api` requests to `http://127.0.0.1:8000`.
The Docker deployment uses Nginx as the same-origin reverse proxy.

## Docker deployment (React + FastAPI)

### Prerequisites
- Docker Engine or Docker Desktop
- Docker Compose v2 (`docker compose version`)

### Start the complete application

Run from the project root (the directory containing `docker-compose.yml`):

```bash
docker compose up --build -d
```

Services:
- **React frontend (primary):** http://localhost:3000
- FastAPI backend: http://localhost:8000
- Interactive API documentation: http://localhost:8000/docs
- API health: http://localhost:8000/health
- **Optional Streamlit frontend:** http://localhost:8501

React is served by Nginx. Nginx forwards `/api/*` requests to FastAPI, so the
browser does not need a separately configured API hostname. The supplied catalog
is mounted read-only into the backend. No API key or external database is needed.

### Verify and inspect logs

```bash
docker compose ps
docker compose logs -f
docker compose logs -f react
docker compose logs -f api
```

### Stop, restart and rebuild

```bash
docker compose down
docker compose up --build -d
```

For a clean rebuild:

```bash
docker compose build --no-cache
docker compose up -d
```

### Test recommendations through the API

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"occasion":"Diwali","limit":8}'
```

### Troubleshooting
- If port 3000, 8000 or 8501 is busy, change the host-side port mapping in
  `docker-compose.yml`.
- React logs: `docker compose logs -f react`
- API logs: `docker compose logs -f api`
- Optional Streamlit logs: `docker compose logs -f streamlit`
- If the frontend loads but recommendations fail, check that the API container is
  running and healthy with `docker compose ps`.


## Groq LLM integration

This version integrates **Groq Chat Completions** with the `openai/gpt-oss-120b`
model. Groq interprets free-form occasion text and expands it into relevant search
phrases and keywords. The backend then combines that expansion with the original
occasion and ranks only real catalog products using TF-IDF cosine similarity and
lexical matching.

**Important design distinction:** Groq is used for LLM-based occasion understanding,
not as an embeddings API. The final recommendation list remains grounded in the
provided `products.json`; the model cannot invent products, prices, images or brands.
The Groq API key is read only by the backend and is never sent to React.

### Configure Groq

1. Create an API key in the [Groq Console](https://console.groq.com/keys).
2. In the project root, copy `.env.example` to `.env`.
3. Put your key in `.env`:

```env
GROQ_API_KEY=gsk_your_real_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

4. Start/rebuild the services:

```bash
docker compose up --build -d
```

Do not commit `.env` to GitHub. `.env` is excluded by `.gitignore`.

### Groq fallback behavior

If `GROQ_API_KEY` is missing, invalid, rate-limited or temporarily unavailable,
the application automatically falls back to its local TF-IDF + occasion-keyword
pipeline. The API health endpoint reports whether Groq is configured:

```bash
curl http://localhost:8000/health
```

The response includes `ai.enabled`, `ai.provider` and the configured model.

### Why this hybrid design?

- **Groq LLM:** understands natural-language occasions such as "a thoughtful gift
  for my colleague who is leaving" and expands the intent into retrieval terms.
- **TF-IDF + cosine similarity:** compares the expanded query with catalog text.
- **Keyword overlap:** preserves exact matches for product names and occasion terms.
- **Catalog constraints:** results are filtered to active, approved, in-stock products.
- **Fallback:** search remains usable when Groq is not configured or unavailable.

Groq API reference: https://console.groq.com/docs/api-reference


### Groq JSON response token budget

The Groq query-expansion request uses JSON-object response mode and a 1,024-token completion budget. This avoids truncating GPT-OSS output before it can finish a valid JSON object. The prompt also constrains the response to a short phrase and at most eight keywords. If Groq still fails or is unavailable, the recommender continues with local occasion expansion and ranking.
