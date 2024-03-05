import pandas as pd    
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA 
from sklearn.metrics import accuracy_score
cancer = pd.read_csv("cancer.csv")
cancer.drop("id", inplace=True, axis=1)
cancer.drop("Unnamed: 32", inplace=True, axis=1)
X = cancer.drop("diagnosis", axis=1)
y = cancer["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)
max = 0.0
maxK = 0
accuracies = []
for i in range(30):
    knn = KNeighborsClassifier(n_neighbors=(i+1), p = 2)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == list(y_test))
    if accuracy > max:
        max = accuracy
        maxK = i+1
    accuracies.append(accuracy)
plt.plot(range(1, 31), accuracies)
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.title("Euclidean Distance Accuracy")
pca = PCA(n_components=2)
X = pca.fit_transform(X_train)
y = y_train.replace('M', 2)
y = y.replace('B', 1)
y = y.to_numpy()
KNN = KNeighborsClassifier(n_neighbors=5, p = 2)
KNN = KNN.fit(X, y)
x_test = pca.fit_transform(X_test)
y_true2 = y_test.replace('M', 2)
y_true2 = y_true2.replace('B', 1)
predicted = KNN.predict(x_test)
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
Z = KNN.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z)

    # Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
print(accuracy_score(y_true2, predicted))
plot_decision_regions(X, y.astype(np.int_), clf = KNN, legend = 2)