import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load('automl_model.joblib')
print('Loaded model  :', model.best_estimator)
print('Test accuracy :', round(accuracy_score(y_test, model.predict(X_test)), 4))
