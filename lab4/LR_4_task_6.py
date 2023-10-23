from sklearn.datasets import _samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

X, Y = _samples_generator.make_classification(n_samples=150, n_features=25, n_classes=3, n_informative=6, n_redundant=0,
                                             random_state=7)

k_best_selector = SelectKBest(f_regression, k=10)

classifier = ExtraTreesClassifier(n_estimators=60, max_depth=4)

pipeline = Pipeline([('selector', k_best_selector), ('erf', classifier)])

pipeline.set_params(selector__k=7, erf__n_estimators=30)

pipeline.fit(X, Y)

output = pipeline.predict(X)
print("Predicted output:", output)

print("Score:", pipeline.score(X, Y))

status = pipeline.named_steps['selector'].get_support()
selected = [i for i, x in enumerate(status) if x]
print("\nIndices of selected features:", ', '.join([str(x) for x in selected]))
