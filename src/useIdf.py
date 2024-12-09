import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from embed import RecommenderType, Result


result = Result()
scaler = StandardScaler()
data = result.get_values(RecommenderType.BERT)[:1000]

numbers = [number for _, number in data]

before = PCA(2).fit_transform(numbers)
after = scaler.fit_transform(before)


# plt.scatter([val for val, _ in before], [val for _, val in before])

plt.scatter([val for val, _ in after], [val for _, val in after])


plt.show()
