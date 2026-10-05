import time
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from flaml import AutoML

# ---------- Task 1: Load and split data ----------
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ---------- Task 2: Baselines ----------
t0 = time.time()
base = LogisticRegression(max_iter=2000).fit(X_train, y_train)
base_time = time.time() - t0
base_acc = accuracy_score(y_test, base.predict(X_test))
print('Baseline accuracy (as in sheet):', round(base_acc, 4),
      f'| train time: {base_time:.2f}s')

# Fairer baseline with feature scaling (for discussion)
base_scaled = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000)).fit(X_train, y_train)
base_scaled_acc = accuracy_score(y_test, base_scaled.predict(X_test))
print('Baseline accuracy (scaled)     :', round(base_scaled_acc, 4))

# ---------- Task 3: Run AutoML ----------
automl = AutoML()
t0 = time.time()
automl.fit(X_train, y_train,
           task='classification',
           metric='accuracy',
           time_budget=60,
           seed=42,
           log_file_name='flaml.log')
auto_time = time.time() - t0

# ---------- Task 4: Inspect the best model ----------
print('\n===== Best model =====')
print('Best estimator :', automl.best_estimator)
print('Best config    :', automl.best_config)
print('Best CV score  :', round(1 - automl.best_loss, 4))

# ---------- Task 5: Evaluate on the test set ----------
auto_acc = accuracy_score(y_test, automl.predict(X_test))
print('\n===== Test set comparison =====')
print('AutoML test accuracy   :', round(auto_acc, 4), f'| search time: {auto_time:.1f}s')
print('Baseline test accuracy :', round(base_acc, 4), f'| train time: {base_time:.2f}s')
diff = auto_acc - base_acc
print(f'Difference             : {diff:+.4f} '
      f'(~{round(diff * len(y_test))} test samples out of {len(y_test)})')

# ---------- Task 6: Save the best model ----------
joblib.dump(automl, 'automl_model.joblib')
print('\nSaved automl_model.joblib')