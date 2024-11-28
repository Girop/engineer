from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
import pickle


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, default='text')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    contents = []
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=0.05, max_features=800)
    files = list(arguments.input.iterdir())
    for file in tqdm(files):
        with open(file, "r") as fp:
            contents.append(fp.read())

    data = vectorizer.fit_transform(contents)

    sparse_df: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(data, columns=vectorizer.get_feature_names_out())
    print(sparse_df.info())

    sparse_df.to_pickle('tf_dataframe.pkl')
    with open("tf_vectorizer.pkl", "wb+") as fp:
        pickle.dump(vectorizer, fp)
