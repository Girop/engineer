from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from dbTypes import EmbeddingData, Embedders, Base, Metadata, Ratings
import gensim
import torch
from transformers import BertModel, BertTokenizer
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import gc
from time import time
from enum import Enum


class RecommenderType(Enum):
    BERT = 1
    GLOVE = 2
    IDF_TF = 3


def parse_embed_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, default="text", help='directory with text files')
    parser.add_argument('-m', '--mode', type=int, required=True)
    return parser.parse_args()


class GloveEmbedder:
    def __init__(self) -> None:
        glove_file = "glove.6B.300d.txt"
        self.__glove_dict = gensim.models.keyedvectors.load_word2vec_format(glove_file, binary=False, no_header=True)

    def get_from_file(self, path: Path) -> np.ndarray:
        return self.get(self.read_doc(path))

    @staticmethod
    def read_doc(filepath: Path) -> list[str]:
        with open(filepath, encoding='utf-8') as fp:
            res = fp.read().lower().strip().split()
        return res

    def get(self, words: list[str]) -> np.ndarray:
        result: list[np.ndarray] = [
            self.__glove_dict[tok] for tok in words
            if tok in self.__glove_dict
        ]
        res = np.mean(result, axis=0)
        return res


class BertEmbedder:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            print("Warning: not running on CUDA")
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', torch_dtype=torch.float16)
        self.__model = BertModel.from_pretrained('bert-base-uncased', torch_dtype=torch.float16).to(self.__device)


    @staticmethod
    def read_doc(filepath: Path) -> list[str]:
        with open(filepath, encoding='utf-8') as fp:
            content = fp.readlines()
        return [line.strip() for line in content if len(line) != 0][:512]


    def get_from_file(self, path: Path) -> np.ndarray:
        return self.get(self.read_doc(path))


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


    def add_rating(self, first: RecommenderType, second: RecommenderType, third: RecommenderType) -> bool:
        try:
            self.__session.add(Ratings(first=first.value, second=second.value, third=third.value))
            self.__session.commit()
            return True
        except Exception as e:
            self.__session.rollback()
            print(f"Failed to add rating: {e}")
            return False

    def get_processed_names(self, embedder_type: RecommenderType) -> list[str]:
        return [
            str(item.fileName) for item in
            self.__session.query(EmbeddingData).filter(EmbeddingData.usedModelId == embedder_type.value).all()
        ]


    def get_values(self, embedder_type: RecommenderType) -> list[tuple[str, np.ndarray]]:
        query = (
            select(EmbeddingData, Embedders)
            .filter(EmbeddingData.usedModelId == embedder_type.value)
            .join(Embedders, EmbeddingData.usedModelId == Embedders.modelId)
        )
        result = self.__session.execute(query).fetchall()
        used_float = np.float16 if embedder_type == RecommenderType.BERT else np.float32

        return [
            (values.fileName, np.frombuffer(values.values, used_float, embedder.shape))
            for values, embedder in result
        ]

    def get_metadata(self, paper_id: str) -> Metadata:
        return self.__session.query(Metadata).filter(Metadata.article_id == paper_id.split('v')[0]).first()

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


class GeneralEmbedder:
    def __init__(self, embedder_type: RecommenderType) -> None:
        self.__embedder_type = embedder_type
        self.embedder = self.__get_embedder(embedder_type)()
        self.saving = Result()
        self.__type = embedder_type
        self.__timings = []

    @staticmethod
    def __get_embedder(embedder_type: RecommenderType):
        if embedder_type == RecommenderType.BERT:
            return BertEmbedder
        elif embedder_type == RecommenderType.GLOVE:
            return GloveEmbedder
        else:
            raise Exception("Unimplemented method")

    def __remove_versioned(self, input_files: set[Path]) -> list[Path]:
        seen: list[Path] = []
        for file in input_files:
            base_identifier = file.stem.split('v')[0]
            if base_identifier not in seen:
                seen.append(file)
        return seen

    def __get_files_to_process(self, in_dir: Path) -> list[Path]:
        processed_stems = self.saving.get_processed_names(self.__embedder_type)
        files = list(in_dir.iterdir())
        previous = [in_dir / (prev_stem + ".txt") for prev_stem in processed_stems]
        skipped = set(files).intersection(set(previous))

        print(f"All files: {len(files)}")
        print(f"Articles to be processed: {len(files) - len(skipped)}, Skipping: {len(skipped)} + versioned")
        return list(set(files) - set(previous))
        # return self.__remove_versioned(set(files) - set(previous))

    @staticmethod
    def __split_text(text: str) -> list[str]:
        return [
                line.strip() for line in
                text.split('\n') if len(line) != 0
        ][:1024]

    def get(self, text: str) -> np.ndarray:
        adjusted_text = self.__split_text(text) if self.__type == RecommenderType.BERT else text.lower().split()
        return self.embedder.get(adjusted_text)

    def run_for_all(self, input_dir: Path):
        filepaths = self.__get_files_to_process(input_dir)
        for filepath in tqdm(filepaths):
            try:
                t1 = time()
                embedding = self.embedder.get_from_file(filepath)
                self.__timings.append(time() - t1)
                self.saving.add(str(filepath.stem), embedding, current_mode.value)
            except Exception as e:
                print(f"Failed: {e}")
            gc.collect()
        self.saving.finish()
        self.__print_timings()

    def __print_timings(self):
        print("\nTimings: \n", self.__timings)


if __name__ == '__main__':
    in_arguments = parse_embed_arguments()
    current_mode = RecommenderType(in_arguments.mode)
    embedder = GeneralEmbedder(current_mode)
    embedder.run_for_all(in_arguments.input)
    print("Finished")
