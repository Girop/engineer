from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
from embed import GeneralEmbedder, Result, EmbedderType
from dbTypes import Metadata
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum
from typing import Optional
import requests


class RecommendationException(Exception):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(message)

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




class ReqType(Enum):
    PDF = 1
    TEXT = 2
    ARXIV_ID = 3


@dataclass
class RecomendationReq:
    req_type: ReqType
    used_model: EmbedderType
    value: str


@dataclass
class Recommendation:
    arxiv_id: str
    similarity_score: float # allowed range [-1, 1]
    title: str
    category: str
    date: str


class Recommender:
    def __init__(self, mode: EmbedderType):
        print("Loading data")
        self.result = Result()
        self.__scaler = StandardScaler()
        self.__embedder = GeneralEmbedder(mode)
        data = self.result.get_values(mode)
        self.__names = [name for name, _ in data]
        numbers = [number for _, number in data]
        self.__data = np.vstack(self.__scaler.fit_transform(numbers))

    def recomend_paper(self, req: RecomendationReq, limit = 10) -> list[Recommendation]:
        if req.req_type != ReqType.ARXIV_ID:
            raise RecommendationException("Unsupported mode")

        if (text := self.__arxiv_id_to_txt(req.value)) is None:
            raise RecommendationException("Article download failed")

        print("Evaluating")
        embedding_values = self.__embedder.get(text)

        # if req.req_type == ReqType.ARXIV_ID and req.value not in self.result.get_processed_names(req.used_model):
        #     print("Saving to db")
        #     self.result.add(req.value, embedding_values, req.used_model.value)

        print("Transforming")
        return self.__recommend(embedding_values, limit)

    def __recommend(self, embedding_values: np.ndarray, limit: int) -> list[Recommendation]:
        scaled = self.__scaler.transform(embedding_values.reshape(1, -1))
        distances = cosine_similarity(self.__data, scaled)
        distance_point = sorted(zip(distances, self.__names), key=lambda x: x[0], reverse=True)
        result = []
        for distance, arxiv_id in distance_point[:limit]:
            meta = self.result.get_metadata(arxiv_id)
            result.append(Recommendation(arxiv_id, distance[0], meta.title, meta.categories, meta.update_date))
        return result


    def __arxiv_id_to_txt(self, id: str) -> Optional[str]:
        print("Downloading")
        res = requests.get(id_to_url(id))
        if not res.ok:
            return None
        return pdf2text(res.content)


if __name__ == '__main__':
    my_id = "2410.24080"
    recomender = Recommender(EmbedderType.GLOVE)
    found = recomender.recomend_paper(RecomendationReq(ReqType.ARXIV_ID, EmbedderType.BERT, my_id))
    print(found)
    recomender.result.finish()
