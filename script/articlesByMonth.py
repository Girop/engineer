from pathlib import Path
import seaborn as sns
import json
import pandas as pd
import matplotlib.pyplot as plt


statistics = Path("stats")
jsons = []

for file in statistics.iterdir():
    with open(file, 'r') as fp:
        jsons.append(json.load(fp))

failed, unreadable, suc = 0,0,0


month_names = [
    'Apr 2007',
    'May 2007',
    'Jun 2007',
    'Jul 2007',
    'Aug 2007',
    'Sep 2007',
    'Oct 2007',
    'Nov 2007',
    'Dec 2007',
    'Jan 2008',
    'Feb 2008',
    'Mar 2008',
    'Apr 2008',
    'May 2008',
    'Jun 2008',
    'Jul 2008',
    'Aug 2008',
    'Sep 2008',
    'Oct 2008',
    'Nov 2008',
    'Dec 2008',
]

print("Months", len(month_names))
print("Data", len(jsons))

counts = []

for stat in jsons:
    failed += stat['failed']
    unreadable += stat['unreadable']
    suc_this = stat['skipped'] + stat['successful']
    counts.append(suc_this)
    suc += suc_this

print(counts)
print("Avg:", sum(counts) / len(counts))
print("Min:", min(counts))
print("Max:", max(counts))
print("First: ", counts[0])


exit()
df = pd.DataFrame({
    "month": month_names,
    "count": counts
})

plt.figure(figsize=(12, 6))

sns.barplot(x='month', y='count', data=df, palette='viridis')

plt.xticks(rotation=45, ha='right')

plt.title('Articles uploaded to arXiv', fontsize=14)
plt.xlabel('Month and year', fontsize=12)
plt.ylabel('Available articles', fontsize=12)

plt.tight_layout()
plt.show()

