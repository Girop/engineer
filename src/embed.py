from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from dbTypes import EmbeddingData, Embedders, Base
import torch
from transformers import BertModel, BertTokenizer
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import gc
from enum import Enum


class EmbedderType(Enum):
    BERT = 1


def parse_embed_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, default="text", help='directory with text files')
    parser.add_argument('-o', '--output', type=Path, default='database')
    parser.add_argument('-l', '--limit', type=int, default=None)
    return parser.parse_args()


def read_doc(filepath: Path) -> list[str]:
    with open(filepath, encoding='utf-8') as fp:
        content = fp.readlines()
    return [line.strip() for line in content if len(line) != 0][:4092]


class Embedder:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            print("Warning: not running on CUDA")
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', torch_dtype=torch.float16)
        self.__model = BertModel.from_pretrained('bert-base-uncased', torch_dtype=torch.float16).to(self.__device)


    def get(self, lines: list[str]) -> np.ndarray:
        chunks = self.__get_tokens(lines)
        embedding = self.__get_embedding(chunks)
        single_embedding = torch.mean(embedding.pooler_output, dim=0)
        return single_embedding.cpu().numpy()


    def __get_tokens(self, lines: list[str]):
        return self.__tokenizer(
            lines,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.__device)

    def __get_embedding(self, chunk):
        with torch.no_grad():
            outputs = self.__model(**chunk)
        return outputs


class Result:
    def __init__(self) -> None:
        self.__engine = create_engine('sqlite:///engineer.db')
        Base.metadata.create_all(self.__engine)
        Session = sessionmaker(bind=self.__engine)
        self.__session = Session()


    def get_processed_names(self) -> list[str]:
        return [str(item.fileName) for item in self.__session.query(EmbeddingData).all()]


    def get_values(self, model_id: int):
        query = (
            select(EmbeddingData, Embedders)
            .join(Embedders, EmbeddingData.usedModelId == Embedders.modelId)
            .filter(EmbeddingData.usedModelId == model_id)
        )
        results = self.__session.execute(query).fetchall()
        return results

    def add(self, filename: str, emebdding: np.ndarray, model_id: int):
        try:
            entry = EmbeddingData(fileName=filename, usedModelId=model_id, values=emebdding.tobytes())
            self.__session.add(entry)
            self.__session.commit()
        except Exception as e:
            self.__session.rollback()
            print(f"Failed to add entry '{filename}': {e}")


    def finish(self):
        self.__session.close()


# Skip files only if current model for existing files is the same as for files in DB
def get_files_to_process(in_dir: Path, processed_stems: list[str]):
    files = list(in_dir.iterdir())
    previous = [in_dir / (prev_stem + ".txt") for prev_stem in processed_stems]
    skipped = set(files).intersection(set(previous))
    print(f"All files: {len(files)}")
    print(f"Articles to be processed: {len(files) - len(skipped)}, skipping: {len(skipped)}")
    return set(files) - set(previous)


# TODO add non-contextual embedder (word2vec) and idf-tf 
if __name__ == '__main__':
    in_arguments = parse_embed_arguments()
    saving = Result()
    embedder = Embedder()
    filepaths = get_files_to_process(in_arguments.input, saving.get_processed_names())
    current_model = EmbedderType.BERT
    gc.collect()

    limit = in_arguments.limit or len(filepaths)
    iter = zip(filepaths, range(limit)) # Limiting iterator
    for file, _ in tqdm(iter, total=limit):
        try:
            embedding = embedder.get(read_doc(file))
            saving.add(str(file.stem), embedding, current_model.value)
        except Exception as e:
            print(f"Failed: {e}")
        gc.collect()
    saving.finish()
    print("Gone right")
