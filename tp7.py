import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
iris = datasets.load()

X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']
print(y)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, n_init = 1, init = 'kmeans++')
kmeans.fit_predict(X)
print(kmeans.labels_)
print(y)

from numpy import np
colormap = np.array(['Red','Green','Blue'])
plt.scatter(X.Sepal_Length, X.Sepal_Width, c=colormap[y.Targets], s=40)
plt.show()
plt.close()

plt.scatter(X.Sepal_Length, X.Sepal_Width, c=colormap[kmeans.labels_], s=40)
plt.show()
plt.close()

print("la coherance", metrics.adjusted_rand_score(model.labels_, list(y['Targets'])))