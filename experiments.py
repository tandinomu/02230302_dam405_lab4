import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from flaml import AutoML

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("--- Effect of time budget (metric = accuracy) ---")
print(f"{'Budget':>7} | {'Best model':>10} | {'CV acc':>6} | {'Test acc':>8} | {'Best found at':>13} | {'Total time':>10}")
for budget in [10, 30, 60, 120]:
    automl = AutoML()
    t0 = time.time()
    automl.fit(X_train, y_train, task='classification', metric='accuracy',
               time_budget=budget, seed=42, verbose=0)
    elapsed = time.time() - t0
    test_acc = accuracy_score(y_test, automl.predict(X_test))
    print(f"{budget:>6}s | {automl.best_estimator:>10} | {1 - automl.best_loss:>6.4f} | "
          f"{test_acc:>8.4f} | {automl.time_to_find_best_model:>12.1f}s | {elapsed:>9.1f}s")

print("\n--- Different metric: roc_auc (budget 60s) ---")
automl = AutoML()
automl.fit(X_train, y_train, task='classification', metric='roc_auc',
           time_budget=60, seed=42, verbose=0)
auto_auc = roc_auc_score(y_test, automl.predict_proba(X_test)[:, 1])

base_scaled = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000)).fit(X_train, y_train)
base_auc = roc_auc_score(y_test, base_scaled.predict_proba(X_test)[:, 1])

print('Best estimator        :', automl.best_estimator)
print('Best CV AUC           :', round(1 - automl.best_loss, 4))
print('AutoML test AUC       :', round(auto_auc, 4))
print('Scaled baseline test AUC:', round(base_auc, 4))
