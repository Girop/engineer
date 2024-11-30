from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path
import io
from embed import GeneralEmbedder, Result, RecommenderType
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum
from typing import Optional
import requests
import pickle
from time import time


class RecommendationException(Exception):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(message)

@dataclass
class Recommendation:
    arxiv_id: str
    similarity_score: float
    title: str
    category: str
    authors: str
    date: str


def pdf2text(bytes_):
    pdf_file = io.BytesIO(bytes_)
    resource_manager = PDFResourceManager()
    text_stream = io.StringIO()
    laparams = LAParams()

    device = TextConverter(resource_manager, text_stream, laparams=laparams)
    interpreter = PDFPageInterpreter(resource_manager, device)
    for page in PDFPage.get_pages(pdf_file):
        interpreter.process_page(page)

    extracted_text = text_stream.getvalue()
    text_stream.close()
    return extracted_text


def id_to_url(id: str) -> str:
    return f"https://arxiv.org/pdf/{id}"


def arxiv_id_to_txt(id: str) -> Optional[str]:
    print("Downloading")
    res = requests.get(id_to_url(id))
    if not res.ok:
        return None
    return pdf2text(res.content)


class ReqType(Enum):
    PDF = 1
    TEXT = 2
    ARXIV_ID = 3


@dataclass
class RecomendationReq:
    req_type: ReqType
    mode: RecommenderType
    content: str
    limit: int


def map_category(raw_category: str) -> str:
    names = [name.split('.')[0] for name in raw_category.split()]
    mapping_to_even_simpler = {
        "hep-ph": "physics",
        "cs": "computer science",
        "math": "mathematics",
        "physics": "physics",
        "cond-mat": "physics",
        "gr-qc": "physics",
        "astro-ph": "physics",
        "hep-th": "physics",
        "hep-ex": "physics",
        "nlin": "physics",
        "q-bio" : "quantitative biology",
        "quant-ph": "physics",
        "nucl-th": "physics",
        "hep-lat": "physics",
        "math-ph": "physics",
        "nucl-ex": "physics",
        "stat": "statistics",
        "q-fin": "quantitative finance",
        "econ": "economics"
    }
    return ", ".join({mapping_to_even_simpler[name] for name in names})


def get_text_from_mode(mode, content):
    if mode == ReqType.TEXT:
        return content

    if mode == ReqType.ARXIV_ID:
        txt = arxiv_id_to_txt(content)
        if txt is None:
            raise RecommendationException("Article download failed")
        return txt

    if mode != ReqType.PDF:
        raise RecommendationException("Unsupported mode")

    return pdf2text(content)


class TfRecommender:

    def __init__(self):
        print("Loading tf")
        self.__df = pd.read_pickle('tf_dataframe.pkl').values
        self.__vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85)
        with open("tf_vectorizer.pkl", "rb") as fp:
            self.__vectorizer = pickle.load(fp)
        self.__names = self.__load_filenames(Path("text"))
        self.result = Result()


    def finish(self):
        self.result.finish()


    @staticmethod
    def __load_filenames(path: Path):
        return [
            item.name.split('v')[0] for item in path.iterdir()
        ]

    def get(self, text: list[str]):
        return self.__vectorizer.transform(text)


    def recommend_papers(self, text: str, req: RecomendationReq) -> list[Recommendation]:
        print("Evaluation itf")
        vec = self.get([text])[0]
        cosines = cosine_similarity(vec, self.__df)[0]
        indicies = np.argsort(cosines)[-req.limit:][::-1]
        res = []
        for index in indicies:
            arxiv_id = self.__names[index]
            meta = self.result.get_metadata(arxiv_id)
            if meta is None:
                continue
            res.append(Recommendation(
                arxiv_id,
                cosines[index],
                str(meta.title),
                map_category(str(meta.categories)),
                str(meta.authors),
                str(meta.update_date)
            ))
        return res


class RecommenderEmb:
    def __init__(self, mode: RecommenderType):
        print("Loading data")
        self.result = Result()
        self.__scaler = StandardScaler()
        self.__embedder = GeneralEmbedder(mode)
        data = self.result.get_values(mode)
        self.__names = [name for name, _ in data]
        numbers = [number for _, number in data]
        self.__data = np.vstack(self.__scaler.fit_transform(numbers))


    def recommend_papers(self, text: str, req: RecomendationReq) -> list[Recommendation]:
        print("Evaluating")
        embedding_values = self.__embedder.get(text)

        print("Transforming")
        return self.__recommend(embedding_values, req.limit)


    def finish(self):
        self.result.finish()


    def __recommend(self, embedding_values: np.ndarray, limit: int, ignore_versioned = True) -> list[Recommendation]:
        scaled = self.__scaler.transform(embedding_values.reshape(1, -1))
        distances = cosine_similarity(self.__data, scaled)
        distance_point = sorted(zip(distances, self.__names), key=lambda x: x[0], reverse=True)
        result = []

        added, index = 0, 0
        seen = set()
        while added < limit:
            distance, arxiv_id = distance_point[index]
            simple_id = arxiv_id.split('v')[0]
            index += 1
            if ignore_versioned and simple_id in seen:
                continue
            added += 1
            seen.add(simple_id)
            meta = self.result.get_metadata(arxiv_id)
            result.append(Recommendation(
                arxiv_id,
                distance[0],
                str(meta.title),
                map_category(str(meta.categories)),
                str(meta.authors),
                str(meta.update_date)
            ))

        return result



class Recommender:
    def __init__(self, method: RecommenderType) -> None:
        if method == RecommenderType.BERT or method == RecommenderType.GLOVE:
            self.__recommender = RecommenderEmb(method)
        else:
            self.__recommender = TfRecommender()

    def recommend(self, req) -> list[Recommendation]:
        text = get_text_from_mode(req.req_type, req.content)
        return self.__recommender.recommend_papers(text, req)

    def finish(self):
        # TODO Timing measurement
        self.__recommender.finish()

if __name__ == '__main__':
    my_id = "2410.24080"
    recomender = Recommender(RecommenderType.BERT)
    # recomender = TfRecommender()
    req = RecomendationReq(ReqType.ARXIV_ID, RecommenderType.BERT, my_id, 10)
    found = recomender.recommend(req)
    print(found)
    recomender.finish()
