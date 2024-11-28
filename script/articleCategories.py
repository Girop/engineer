import sys
import os

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)


from pathlib import Path
from embed import Result
from tqdm import tqdm
from collections import defaultdict
import json


res = Result()

categories = defaultdict(lambda: 0)

for transcription in tqdm(list(Path("text").iterdir())):
    short_id = transcription.name.split('v')[0]
    category = res.get_metadata(short_id).categories
    categories[category] += 1


with open("categories.json", "w+") as fp:
    json.dump(dict(categories), fp)

