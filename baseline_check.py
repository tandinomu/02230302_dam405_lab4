from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

base = LogisticRegression(max_iter=2000).fit(X_train, y_train)
print('Baseline accuracy (as in sheet):', round(accuracy_score(y_test, base.predict(X_test)), 4))

base_scaled = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000)).fit(X_train, y_train)
print('Baseline accuracy (scaled)     :', round(accuracy_score(y_test, base_scaled.predict(X_test)), 4))
