from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from embed import EmbedderType
from findPapers import *
import dataclasses

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

global_bert = RecommenderEmb(EmbedderType.BERT)
global_glove = RecommenderEmb(EmbedderType.GLOVE)
global_idf = TfRecommender()


def get_method(mode: EmbedderType):
    if mode == EmbedderType.GLOVE:
        return global_glove
    if mode == EmbedderType.BERT:
        return global_bert
    return global_idf


def server_recommend(req: RecomendationReq, limit: int):
    embedder = get_method(req.mode)
    return [
        dataclasses.asdict(item) for item in
        embedder.recommend_paper(req, limit)
    ]


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.get('/recommend/')
async def recommend(document_id: str, mode: int, limit: int):
    avalible_modes = [1,2,3]
    result = []
    if document_id == '':
        return {'error': 'No document provided', 'result': result}

    if mode not in avalible_modes:
        return {'error': 'Invalid mode provided', 'result': result}

    req = RecomendationReq(ReqType.ARXIV_ID, EmbedderType(mode), document_id)
    try:
        result.extend(server_recommend(req, min(limit, 25)))
    except RecommendationException as e:
        return {'error': f'Recommendation failed: {e.message}', 'result': result}
    return {'error': 'None', 'result': result}
