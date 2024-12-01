from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from numpy import argsort
from embed import Result
from findPapers import (
    Recommender,
    RecommenderType,
    RecomendationReq,
    RecommendationException,
    ReqType
)
import dataclasses

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

global_bert = Recommender(RecommenderType.BERT)
global_glove = Recommender(RecommenderType.GLOVE)
global_idf = Recommender(RecommenderType.IDF_TF)


def get_method(mode: RecommenderType):
    if mode == RecommenderType.GLOVE:
        return global_glove
    if mode == RecommenderType.BERT:
        return global_bert
    return global_idf


def server_recommend(req: RecomendationReq):
    recommender = get_method(req.mode)
    return [
        dataclasses.asdict(item) for item in
        recommender.recommend(req)
    ]


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


def recommend_inner(mode, limit, req_type, payload):
    avalible_modes = [1,2,3]
    result = []

    if mode not in avalible_modes:
        return {'error': 'Invalid mode provided', 'result': result}

    if req_type not in [1,2,3]:
        return {'error': 'Invalid req type provided', 'result': result}

    try:
        req = RecomendationReq(
            ReqType(req_type),
            RecommenderType(mode),
            payload,
            min(limit, 25)
        )
        result.extend(server_recommend(req))
    except RecommendationException as e:
        return {'error': f'Recommendation failed: {e.message}', 'result': result}
    return {'error': 'None', 'result': result}


@app.post('/recommend/pdf/')
async def recommend_pdf(
    mode: int = Form(...),
    limit: int = Form(...),
    req_type: int = Form(...),
    payload: UploadFile = File(None)
):
    pdf_file = await payload.read()
    return recommend_inner(mode, limit, req_type, pdf_file)


@app.post('/recommend/')
async def recommend(
    mode: int = Form(...),
    limit: int = Form(...),
    req_type: int = Form(...),
    payload: str = Form(...)
):
    if (len(payload)) <= 2:
        return {"error": "Too short input", 'result': []}
    return recommend_inner(mode, limit, req_type, payload)


class RatingRequest(BaseModel):
    bert: int
    glove: int
    tf: int


@app.post('/rating/')
async def rating(rating: RatingRequest):
    res = Result()
    # This has to be done this weird way around because *relations* in the db
    rankings = [rating.bert, rating.glove, rating.tf]
    recommenders = [RecommenderType.BERT, RecommenderType.GLOVE, RecommenderType.IDF_TF]
    elements = {"first": 0, "second": 0, "third": 0}

    for index, key in zip(argsort(rankings), elements.keys()):
        elements[key] = recommenders[index]

    if not res.add_rating(**elements):
        print("Rating collection failed")

    return {"message": "rating added"}
