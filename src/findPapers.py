from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
from embed import Result, EmbedderType
from dbTypes import Embedders, EmbeddingData
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from enum import Enum
from typing import Union
from embed import Embedder
import requests


def load_data(res: Result):
    values = res.get_values(EmbedderType.BERT.value)

    embeddings: list[np.ndarray] = []
    names: list[str] = []

    for value in values:
        data: EmbeddingData = value[0]
        embedder: Embedders = value[1]
        embeddings.append(np.frombuffer(data.values, np.float16, embedder.shape))
        names.append(data.fileName)
    return embeddings, names


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


def split_text(text: str) -> list[str]:
    return [
            line.strip() for line in
            text.split('\n') if len(line) != 0
    ][:4092]


class ReqType(Enum):
    PDF = 1
    TEXT = 2
    ARXIV_ID = 3


@dataclass
class RecomendationReq:
    reqType: ReqType
    value: Union[bytes, str]


@dataclass
class Recommendation:
    arxiv_id: str
    similarity_score: float # allowed range [-1, 1]
    distance: float


class Recommender:
    def __init__(self):
        print("Loading data")
        self.result = Result()
        embeddings, names = load_data(self.result)
        self.__scaler = StandardScaler()
        self.__data_stacked = np.vstack(self.__scaler.fit_transform(embeddings))
        self.__names = names
        self.__embedder = Embedder()


    def recomend_paper(self, req: RecomendationReq, limit = 10) -> list[Recommendation]:
        if req.reqType != ReqType.ARXIV_ID:
            print("TODO: othere sources")
            exit()

        text = self.__arxiv_id_to_txt(req.value)
        if text is None:
            print("Failed something")
            exit()

        print("Evaluating")
        reference_article = self.__embedder.get(text)

        if req.reqType == ReqType.ARXIV_ID and req.value not in self.__names:
            # TODO validate that the article under this ID was downloaded & don't evaluate it if it already exists in db
            print("Saving to db")
            self.result.add(req.value, reference_article, 1)

        print("Transforming")
        scaled = self.__scaler.transform(reference_article.reshape(1, -1))
        distances = cosine_similarity(self.__data_stacked, scaled)
        named_points = sorted(zip(distances, self.__names), key=lambda x: x[0], reverse=True)
        return [
            Recommendation(point[1], point[0], 0) for point in named_points[:limit]
        ]


    def __arxiv_id_to_txt(self, id: str) -> list[str]:
        print("Downloading")
        res = requests.get(id_to_url(id))
        if not res.ok:
            print("Failed request")
            return []
        return split_text(pdf2text(res.content))


if __name__ == '__main__':
    my_id = "2410.24080"
    recomender = Recommender()
    found = recomender.recomend_paper(RecomendationReq(ReqType.ARXIV_ID, my_id))
    for rec in found:
        print(rec)
    recomender.result.finish()
