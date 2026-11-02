from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from app.recommender import OccasionRecommender

app = FastAPI(title="Occasion Product Recommendation API", version="1.0.0", description="Hybrid Sentence Transformers, TF-IDF and keyword product recommendations with optional Groq query expansion.")
engine = OccasionRecommender()

class RecommendationRequest(BaseModel):
    occasion: str = Field(min_length=1, max_length=120, examples=["Diwali"])
    limit: int = Field(default=12, ge=1, le=50)

@app.get("/health")
def health():
    return {"status": "ok", "catalog": engine.catalog_stats, "ai": engine.ai_status}

@app.get("/products")
def products(limit: int = Query(50, ge=1, le=200)):
    return {"count": min(limit, len(engine.products)), "total": len(engine.products), "products": engine.products[:limit]}

@app.post("/recommendations")
def recommendations(payload: RecommendationRequest):
    results = engine.recommend(payload.occasion, payload.limit)
    return {"occasion": payload.occasion, "count": len(results), "recommendations": results}

@app.get("/products/{product_id}")
def product_detail(product_id: str):
    product = engine.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product
