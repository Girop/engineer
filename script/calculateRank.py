import sys
import os

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)

from dbTypes import Ratings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from collections import defaultdict
from embed import RecommenderType
from scipy.stats import wilcoxon, variation
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


engine = create_engine('sqlite:///engineer.db')
Session = sessionmaker(bind=engine)
session = Session()

ratings = list(session.query(Ratings).all())

ranks = defaultdict(list)


for arr in ratings:
    for i, elem in enumerate([arr.first, arr.second, arr.third]):
        ranks[RecommenderType(elem)].append(i+1)


stat, p = wilcoxon(ranks[RecommenderType.IDF_TF], ranks[RecommenderType.BERT])

print(p)
print(stat)
exit()

for key, rank_collection in ranks.items():
    print(f"Key = {key}")
    median = sorted(rank_collection)
    meidan = (median[len(median) // 2] + median[len(median) // 2 + 1]) / 2
    mean = sum(rank_collection) / len(rank_collection)
    var = variation(rank_collection)
    print(f"Median = {meidan}, Mean = {mean}, var = {var}")


plot_data = { "method": [], "rank": [] }


for key, items in ranks.items():
    name = ""
    if key == RecommenderType.GLOVE:
        name = "GloVe"
    elif key == RecommenderType.IDF_TF:
        name = "TDF-TF"
    else:
        name = "BERT"

    plot_data['method'].extend([name] * len(items))
    plot_data['rank'].extend(items)


exit()

df = pd.DataFrame(plot_data)

plt.figure(figsize=(8, 6))
sns.violinplot(x="method", y="rank", data=df, inner="box", palette="muted")

plt.xlabel("Method", fontsize=12)
plt.ylabel("Rank", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()
