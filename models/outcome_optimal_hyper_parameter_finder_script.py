from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

from data_processing.ml_data_prepairer import get_ml_data
from models.preprocess_pipeline import CongestiveHeartFailurePreprocessor

data = get_ml_data()

preprocessor = CongestiveHeartFailurePreprocessor()
preprocessor.fit(data)
transformed_data = preprocessor.transform(data)

y = data.died.values

X_train, X_test, y_train, y_test = train_test_split(transformed_data, y, test_size=0.3)

pipeline = Pipeline([
    ('rf', RandomForestClassifier())
])

param_grid = [{
    'rf__max_depth': list(range(9, 20)),
    'rf__n_estimators': list(range(45, 70, 5)),
    'rf__criterion': ["gini", "entropy"],
    "rf__max_features": ["auto", None]
}]

gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy')

gs.fit(X_train, y_train)

print("Best hyper parameters")
print(gs.best_params_)
# {'rf__max_depth': 19, 'rf__n_estimators': 55, 'rf__criterion': 'entropy', 'rf__max_features': None}

clf = gs.best_estimator_
clf.fit(X_train, y_train)

print("Train accuracy: %.3f" % clf.score(X_train, y_train))
print("Test accuracy: %.3f" % clf.score(X_test, y_test))