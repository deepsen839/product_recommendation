import os
os.environ["ENABLE_SENTENCE_TRANSFORMERS"] = "false"

from app.recommender import OccasionRecommender


def test_catalog_loads_and_filters():
    engine = OccasionRecommender()
    assert engine.catalog_stats["total"] == 110
    assert engine.catalog_stats["eligible"] == 104
    assert all(p["status"] == "ACTIVE" and p["isApproved"] is True and p["stock"] > 0 for p in engine.products)


def test_diwali_returns_relevant_products():
    engine = OccasionRecommender()
    results = engine.recommend("Diwali", limit=10)
    assert results
    assert any("diwali" in (p.get("name", "") + p.get("description", "")).lower() for p in results)


def test_empty_occasion():
    engine = OccasionRecommender()
    assert engine.recommend("   ") == []


def test_limit_is_respected():
    engine = OccasionRecommender()
    assert len(engine.recommend("birthday", limit=3)) <= 3


def test_api_health_and_recommendations():
    from fastapi.testclient import TestClient
    from app.api import app
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["catalog"]["eligible"] == 104
    response = client.post("/recommendations", json={"occasion": "Diwali", "limit": 3})
    assert response.status_code == 200
    assert response.json()["count"] <= 3
    assert response.json()["recommendations"]
