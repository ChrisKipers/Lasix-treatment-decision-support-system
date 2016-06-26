from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

from data_processing.ml_data_prepairer import get_ml_data
from models.preprocess_pipeline import CongestiveHeartFailurePreprocessor

data = get_ml_data()

treatment_label_binarizer = LabelBinarizer()

preprocessor = CongestiveHeartFailurePreprocessor(False)
preprocessor.fit(data)
transformed_data = preprocessor.transform(data)

y = treatment_label_binarizer.fit_transform(data.treatment.values)

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

gs.fit(transformed_data, y)

print("Best hyper parameters")
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(X_train, y_train)

print("Train score: %.3f" % clf.score(X_train, y_train))
print("Test score: %.3f" % clf.score(X_test, y_test))