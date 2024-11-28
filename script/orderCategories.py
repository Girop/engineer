import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns



with open("categories.json", "r") as fp:
    data = json.load(fp)


simple_mapping = defaultdict(lambda: 0)


for key, value in data.items():
    short_names = set(name.split('.')[0] for name in key.split())

    for name in short_names:
        simple_mapping[name] += value


sum_ = 0
for name, count in simple_mapping.items():
    sum_ += count
    print(f"{name} - {count}")


print("Count", sum_)


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

better_summary = defaultdict(lambda: 0)

for name, count in simple_mapping.items():
    key = mapping_to_even_simpler[name]
    better_summary[key] += count

print("\n\nBetter summary\n\n")
for name, count in better_summary.items():
    print(f"{name} - {count}")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 8))
colors = sns.color_palette("pastel", len(better_summary))

wedges, texts, autotexts = plt.pie(
    list(better_summary.values()),
    labels=None,
    autopct= lambda x: f'{x:.1f}%' if x > 0.1 else "",
    startangle=140,
    pctdistance=1.05,
    colors=colors
)


plt.legend(
    wedges,
    list(better_summary.keys()),
    title="Categories",
    loc="center left",  # Position the legend to the left of the plot
    bbox_to_anchor=(1, 0.5),  # Adjust position: (x, y) of legend anchor point
    fontsize=10
)

plt.title("Proportion of articles across categories", fontsize=14)

plt.show()
