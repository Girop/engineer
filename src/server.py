from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from findPapers import RecommendationException, Recommender, RecomendationReq, ReqType
import dataclasses


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.get('/recommend/id/{document_id}/{mode}')
async def recommend(document_id: str, mode: int):
    avalible_modes = [1,2,3]
    limit = 10
    result = []
    if document_id == '':
        return {'error': 'No document provided', 'result': result}

    if mode not in avalible_modes:
        return {'error': 'Invalid mode provided', 'result': result}

    req = RecomendationReq(ReqType.ARXIV_ID, document_id)
    try:
        recommender = Recommender()
    except RecommendationException as e:
        return {'error': f'Recommendation failed: {e.message}', 'result': result}


    result.extend([
        dataclasses.asdict(item) for item in
        recommender.recomend_paper(req)[:limit]
    ])
    recommender.result.finish()
    return {'error': 'None', 'result': result}
