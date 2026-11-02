"""Hybrid occasion-to-product recommendation engine.

Uses TF-IDF cosine similarity plus transparent occasion keyword boosts.
Only approved, active, non-deleted and in-stock products are recommendable.
"""
from __future__ import annotations
import json, re, os, logging
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = BASE_DIR / "data" / "products.json"

OCCASION_TERMS = {
    "birthday": "birthday party celebration cake balloons gifts surprise friends family",
    "wedding": "wedding marriage bride groom bridal ceremony reception flowers bouquet elegant gifts",
    "anniversary": "anniversary romantic love couple celebration roses flowers perfume chocolate watch",
    "valentine": "valentine valentines day romantic love partner girlfriend boyfriend roses flowers chocolate perfume",
    "diwali": "diwali deepavali festival festive lights prosperity puja sweets dry fruits hamper marigold gifting",
    "office party": "office party corporate event colleagues employees celebration professional gifts self care",
    "housewarming": "housewarming new home home decor lifestyle sustainable gifts",
    "engagement": "engagement ring couple celebration flowers romantic gifts",
    "graduation": "graduation achievement success congratulations professional gifts celebration",
    "farewell": "farewell goodbye colleagues appreciation corporate gifts self care",
    "christmas": "christmas festive holiday gifts celebration decorations chocolates hamper",
    "mothers day": "mothers day mom appreciation flowers perfume chocolates self care gifts",
    "fathers day": "fathers day dad appreciation watch gifts celebration",
    "raksha bandhan": "rakhi raksha bandhan siblings brother sister sweets chocolates gifts",
    "holi": "holi festival colors celebration festive gifting",
    "new year": "new year celebration party festive gifts chocolates",
    "thank you": "thank you appreciation gratitude thoughtful gifts flowers chocolates",
}

CATEGORY_RULES = [
    ("Flowers & Bouquets", ["flower", "rose", "lily", "lilies", "tulip", "marigold", "bouquet", "blossom"]),
    ("Chocolates & Sweets", ["chocolate", "cocoa", "sweet", "candy"]),
    ("Gift Hampers & Packaging", ["hamper", "gift wrapping", "wrapping", "gift box", "dry fruit"]),
    ("Fashion & Accessories", ["watch", "t-shirt", "shirt", "clothes", "jewellery", "jewelry"]),
    ("Fragrance & Beauty", ["perfume", "fragrance", "eau de parfum"]),
    ("Wellness & Lifestyle", ["self-care", "self care", "fitness", "bamboo", "lifestyle", "health"]),
    ("Prints & Personalised Gifts", ["print", "printed", "personalised", "personalized"]),
]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def infer_category(product: dict) -> str:
    text = " ".join(_text(product.get(k)) for k in ("name", "description", "shortDescription", "slug")).lower()
    for label, terms in CATEGORY_RULES:
        if any(term in text for term in terms):
            return label
    return "Gifts & Other"


class OccasionRecommender:
    def __init__(self, catalog_path: str | Path = DEFAULT_CATALOG):
        self.catalog_path = Path(catalog_path)
        self.all_products = self._load()
        self.products = [p for p in self.all_products if self._eligible(p)]
        self.documents = [self._document(p) for p in self.products]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.documents) if self.documents else None
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self._groq_client = None
        self.embedding_enabled = os.getenv("ENABLE_SENTENCE_TRANSFORMERS", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.embedding_model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._embedding_model = None
        self._product_embeddings = None
        self._embedding_attempted = False

    def _load(self) -> list[dict]:
        with self.catalog_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("products", data.get("data", []))
        if not isinstance(data, list):
            raise ValueError("Product catalog must be a JSON array or an object containing products/data.")
        return [p for p in data if isinstance(p, dict)]

    @staticmethod
    def _eligible(p: dict) -> bool:
        status = _text(p.get("status")).upper()
        approved = p.get("isApproved")
        try:
            stock = float(p.get("stock") or 0)
        except (TypeError, ValueError):
            stock = 0
        return status == "ACTIVE" and approved is True and not p.get("deletedAt") and stock > 0

    @staticmethod
    def _document(p: dict) -> str:
        return " ".join(_text(p.get(k)) for k in ("name", "shortDescription", "description", "brand", "slug", "categoryId"))

    @property
    def catalog_stats(self) -> dict:
        return {"total": len(self.all_products), "eligible": len(self.products), "excluded": len(self.all_products)-len(self.products)}

    def _get_groq_client(self):
        """Create the Groq client lazily; never expose the API key to the frontend."""
        if not self.groq_api_key:
            return None
        if self._groq_client is None:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key, timeout=12.0, max_retries=1)
            except Exception as exc:
                logger.warning("Groq client initialization failed: %s", exc)
                return None
        return self._groq_client

    def _expand_with_groq(self, occasion: str) -> str:
        """Use Groq to interpret free-form occasion intent and produce search terms.

        Groq only expands the query. Product facts and final scores always come
        from the supplied catalog and deterministic local ranking.
        """
        client = self._get_groq_client()
        if client is None:
            return ""
        try:
            completion = client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an occasion-based product search query interpreter. "
                            "Return one compact, valid JSON object with exactly two keys: "
                            "'expanded_query' (a short search phrase, maximum 35 words) and "
                            "'keywords' (an array of at most 8 short strings). "
                            "Do not include markdown fences or any text outside the JSON. "
                            "Do not recommend or invent products, brands, prices, or catalog facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Interpret this gift-shopping occasion and expand it for product retrieval: {occasion}",
                    },
                ],
                temperature=0.0,
                # GPT-OSS may spend completion tokens on reasoning before emitting JSON.
                # 180 was too small and caused json_validate_failed on otherwise valid requests.
                max_completion_tokens=1024,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            print(content)
            payload = json.loads(content)
            expanded = _text(payload.get("expanded_query"))
            keywords = payload.get("keywords", [])
            if isinstance(keywords, list):
                expanded += " " + " ".join(_text(item) for item in keywords[:8])
            if not expanded.strip():
                return ""
            return expanded.strip()
        except Exception as exc:
            logger.warning("Groq occasion expansion unavailable; using local expansion: %s", exc)
            return ""

    def _ensure_embeddings(self) -> bool:
        """Load a CPU SentenceTransformer and encode the catalog once.

        Loading is lazy so API startup and TF-IDF fallback remain available if
        the model cannot be downloaded or the optional dependency is missing.
        """
        if not self.embedding_enabled:
            return False
        if self._product_embeddings is not None:
            return True
        if self._embedding_attempted:
            return False
        self._embedding_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")
            self._product_embeddings = self._embedding_model.encode(
                self.documents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            logger.info("Loaded SentenceTransformer %s for %d products", self.embedding_model_name, len(self.documents))
            return True
        except Exception as exc:
            logger.warning("Sentence Transformers unavailable; using TF-IDF fallback: %s", exc)
            self._embedding_model = None
            self._product_embeddings = None
            return False

    @property
    def ai_status(self) -> dict:
        return {
            "provider": "Groq",
            "enabled": bool(self.groq_api_key),
            "model": self.groq_model if self.groq_api_key else None,
            "embeddings": {
                "provider": "Hugging Face Sentence Transformers",
                "enabled": self.embedding_enabled,
                "model": self.embedding_model_name if self.embedding_enabled else None,
                "loaded": self._product_embeddings is not None,
            },
            "fallback": "TF-IDF + occasion keyword expansion",
        }

    def recommend(self, occasion: str, limit: int = 12, min_score: float = 0.0) -> list[dict]:
        occasion = re.sub(r"\s+", " ", _text(occasion)).strip()
        if not occasion:
            return []
        if not self.products or self.matrix is None:
            return []
        expanded = occasion.lower()
        groq_expansion = self._expand_with_groq(occasion)
        if groq_expansion:
            expanded += " " + groq_expansion.lower()
        for key, terms in OCCASION_TERMS.items():
            if key in expanded or expanded in key:
                expanded += " " + terms
        query = self.vectorizer.transform([expanded])
        tfidf_scores = cosine_similarity(query, self.matrix).ravel()

        # Sentence Transformers add semantic similarity beyond exact word overlap.
        # Normalized embeddings make the dot product equivalent to cosine similarity.
        embedding_scores = None
        if self._ensure_embeddings():
            query_embedding = self._embedding_model.encode(
                [expanded], convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            embedding_scores = np.clip(self._product_embeddings @ query_embedding, 0.0, 1.0)

        tokens = set(re.findall(r"[a-z0-9]+", expanded.lower()))
        results = []
        for idx, (p, tfidf_sim) in enumerate(zip(self.products, tfidf_scores)):
            doc = self.documents[idx].lower()
            direct = sum(1 for t in tokens if len(t) > 2 and t in doc)
            lexical = min(direct / max(len(tokens), 1), 1.0)
            if embedding_scores is not None:
                score = float(0.60 * embedding_scores[idx] + 0.25 * tfidf_sim + 0.15 * lexical)
                method = "Groq + Sentence Transformers + TF-IDF + keyword hybrid" if groq_expansion else "Sentence Transformers + TF-IDF + keyword hybrid"
            else:
                score = float(0.78 * tfidf_sim + 0.22 * lexical)
                method = "Groq + TF-IDF + keyword hybrid" if groq_expansion else "TF-IDF + keyword hybrid"
            if score < min_score:
                continue
            result = dict(p)
            result["category"] = infer_category(p)
            result["relevance_score"] = round(score * 100, 1)
            result["match_reason"] = self._reason(p, occasion)
            result["ranking_method"] = method
            results.append(result)
        results.sort(key=lambda x: (-x["relevance_score"], _text(x.get("name")).lower()))
        return results[:max(1, min(int(limit), 50))]

    @staticmethod
    def _reason(p: dict, occasion: str) -> str:
        name = _text(p.get("name")) or "This product"
        category = infer_category(p)
        desc = _text(p.get("shortDescription")) or _text(p.get("description"))
        if desc:
            sentence = re.split(r"(?<=[.!?])\s+", desc)[0]
            return f"{name} is a {category.lower()} option; {sentence[:130]}"
        return f"{name} is listed under {category.lower()} and may suit {occasion}."

    def get_product(self, product_id: str) -> dict | None:
        for p in self.all_products:
            if _text(p.get("id")) == str(product_id):
                result = dict(p)
                result["category"] = infer_category(p)
                return result
        return None
