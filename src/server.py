from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from embed import EmbedderType
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

global_bert = Recommender(EmbedderType.BERT)
global_glove = Recommender(EmbedderType.GLOVE)

def server_recommend(mode, req, limit):
    get_papers = lambda r: r.recomend_paper(req)[:limit + 20]
    seen = set()
    res = []
    for item in get_papers(global_bert if mode == EmbedderType.BERT else global_glove):
        if (simple_id := item.arxiv_id.split('v')[0]) not in seen:
            res.append(dataclasses.asdict(item))
            seen.add(simple_id)
    return res

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

    req = RecomendationReq(ReqType.ARXIV_ID, EmbedderType(mode), document_id)
    try:
        result.extend(server_recommend(mode, req, limit))

    except RecommendationException as e:
        return {'error': f'Recommendation failed: {e.message}', 'result': result}
    return {'error': 'None', 'result': result}
