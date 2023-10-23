import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

housing_data = datasets.load_boston()
X, Y = shuffle(housing_data.data, housing_data.target, random_state=7)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

regressor = AdaBoostRegressor(DecisionTreeClassifier(max_depth=4), n_estimators=400, random_state=7)
regressor.fit(X_train, Y_train)

Y_train_pred = regressor.predict(X_train)
mse = mean_squared_error(Y_train, Y_train_pred)
evs = explained_variance_score(Y_train, Y_train_pred)
print("ADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

feature_importance = regressor.feature_importances_
feature_names = housing_data.feature_names

feature_importance = 100.0 * (feature_importance / max(feature_importance))

index_sorted = np.flipud(np.argsort(feature_importance))
pos = np.arange(index_sorted.shape[0]) + .5

plt.figure()
plt.bar(pos, feature_importance[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted])
plt.ylabel('Relative Importance')
plt.title('Variable Importance')
plt.show()