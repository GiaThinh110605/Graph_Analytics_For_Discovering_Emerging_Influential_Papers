import os
import time
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.recommender import TwoStageHybridRecommender

app = FastAPI(title="CitationLens API", description="Two-Stage Hybrid Scientific Paper Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = TwoStageHybridRecommender()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.get("/api/search")
async def search_papers(q: str = Query(..., min_length=2)):
    results = engine.search_by_title(q, limit=5)
    return {"query": q, "results": results}

@app.get("/api/recommend")
async def get_recommendations(
    paper_id: int = Query(...),
    alpha: float = Query(0.4, ge=0.0, le=1.0),
    top_k: int = Query(10, ge=3, le=50)
):
    t0 = time.time()
    seed, ranked_df = engine.recommend(paper_id=paper_id, alpha=alpha, top_k=top_k)
    if not seed:
        return JSONResponse(status_code=404, content={"error": f"Không tìm thấy Paper ID #{paper_id}"})
    
    latency_ms = round((time.time() - t0) * 1000, 2)
    
    return {
        "status": "success",
        "latency_ms": latency_ms,
        "seed_paper": seed,
        "recommendations": ranked_df.to_dict(orient="records")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
