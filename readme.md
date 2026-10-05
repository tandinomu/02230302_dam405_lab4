# Practical 4 : FLAML AutoML and Unit V

Sep 30, 2026 · Tandin Om

I used FLAML (an open-source AutoML tool) to find and tune a classifier on the breast cancer dataset, compared it with a logistic regression baseline, and saved the model. A properly scaled baseline did as well as AutoML, so AutoML is useful but must be checked against a good baseline.

## How the Practical Links to Unit V

The practical mainly covers 5.1, 5.3.2 and 5.4, and touches 5.5.

| Unit V topic | Where it shows up in my practical |
| --- | --- |
| **5.1 AutoML concepts** | FLAML automated algorithm choice, hyperparameter tuning and cross-validation. I still made the human decisions: dataset, split, metric, time budget and a fair baseline. This is the "Industrial Revolution" idea: machines do the repetitive work, humans do the judgement. |
| **5.3.2 Open-source AutoML** | FLAML is an open-source AutoML library from Microsoft, known for being fast and low-cost. It tried XGBoost, LightGBM, random forest, logistic regression and SGD. |
| **5.3.1 Cloud AutoML** | Not used directly. FLAML is the free, local alternative to tools like AWS SageMaker Autopilot or Azure Automated ML. |
| **5.4 KaizenML** | I measured against a baseline, noticed it did not converge, improved it with scaling, then ran small experiments (time budgets, roc\_auc). This is continuous improvement: measure, change one thing, check again (PDCA). |
| **5.5 Explainability** | Partly: I inspected the best model and its hyperparameters. A next step would be SHAP or feature importance to explain *why* it predicts what it does. |
| **5.2 Feature stores** | Not covered; the dataset was a ready-made table with no feature pipeline. |

## How I Did the Practical (Task by Task)

Everything ran inside a virtual environment on my MacBook Air (Python 3.14.7, FLAML 2.7.0). The main script is `practical4.py`.

### Task 1: Install and load data

I made a project folder and a virtual environment, installed the packages, and saved the exact versions to `requirements.txt`.

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install "flaml[automl]" scikit-learn joblib
pip freeze > requirements.txt
```

Then I loaded the breast cancer dataset (569 samples, 30 features) and split it 80/20 into 455 training and 114 test samples, with `random_state=42`.

![Figure 1: Package installation and version check.](screenshots/2.png)

*Figure 1: Package installation and version check.*

![Figure 2: The script practical4.py in VS Code.](screenshots/3.png)

*Figure 2: The script practical4.py in VS Code.*

### Task 2: Build a baseline

I trained logistic regression (`max_iter=2000`) as the reference model. It gave a convergence warning, which means it did not finish training properly, so I also trained a scaled version with `StandardScaler` for a fair comparison.

- Baseline as in the sheet (unscaled): **0.9561**
- Baseline with scaling: **0.9737**

![Figure 3: Unscaled and scaled baseline accuracies, with the convergence warning.](screenshots/5.png)

*Figure 3: Unscaled and scaled baseline accuracies, with the convergence warning.*

### Task 3: Run AutoML

I ran FLAML with `task='classification'`, `metric='accuracy'`, `time_budget=60` and `seed=42` so the results can be repeated. In 60 seconds it tested over 700 configurations of XGBoost, LightGBM, random forest, logistic regression and SGD.

### Task 4: Inspect the best model

- **Best estimator:** XGBoost
- **Best config:** 43 trees, 7 leaves, learning rate about 0.59, subsample about 0.76
- **Best CV score:** 0.978
- FLAML found this model after about 15.7 seconds.

### Task 5: Evaluate on the test set

| Model | Test accuracy | Correct / 114 | Time |
| --- | --- | --- | --- |
| Baseline (scaled) | 0.9737 | 111 | under 1 s |
| AutoML (XGBoost) | 0.9649 | 110 | 60 s |
| Baseline (unscaled, as in sheet) | 0.9561 | 109 | 0.09 s |

AutoML beat the sheet's baseline by one test sample, but the scaled baseline beat AutoML by one sample.

![Figure 4: Best model, its hyperparameters (Task 4) and the test set comparison (Task 5).](screenshots/4.png)

*Figure 4: Best model, its hyperparameters (Task 4) and the test set comparison (Task 5).*

### Task 6: Save the best model

I saved the model with joblib as `automl_model.joblib`, then reloaded it in a separate script, `load_model.py`. It gave the same result (XGBoost, 0.9649), which proves it saved correctly.

![Figure 5: Reloading the saved model.](screenshots/6.png)

*Figure 5: Reloading the saved model.*

### Optional: Time budget and metric experiment

I reran FLAML with 10, 30, 60 and 120 second budgets, and once with `roc_auc`, in `experiments.py`.

- **Time budget:** 10 s scored lower (0.9561). From 30 s up, the same model was found at about 14 s, so extra time only added cost.
- **roc\_auc:** FLAML chose LightGBM (test AUC 0.9934); the scaled baseline was still higher (0.9974).
- **Main lesson:** On a small, clean dataset a well-built simple model can match AutoML much faster.

![Figure 6: Time budget and ROC-AUC experiment results.](screenshots/7.png)

*Figure 6: Time budget and ROC-AUC experiment results.*

## Reflection Questions (from the practical sheet)

**1. Did AutoML beat your baseline? By how much, and was the extra compute time worth it?** Yes, it beat the sheet's baseline by 0.0088 (0.9649 vs 0.9561), which is just one extra correct prediction out of 114. But that baseline did not converge. The scaled baseline scored 0.9737 and beat AutoML. AutoML took 60 s against under 1 s, so the extra compute was not worth it for this dataset.

**2. What did AutoML automate for you, and which decisions still required a human?** AutoML automated choosing the algorithm, tuning hyperparameters and cross-validating each model; it tested over 700 configurations across XGBoost, LightGBM, random forest, logistic regression and SGD. I still had to choose the dataset, split, metric and time budget, build a fair baseline, spot that it needed scaling, and judge whether the final model was good enough.

**3. How does increasing the time budget affect the result and the cost?** Cost grows directly with the budget, but the result does not keep improving. At 10 s the search was cut short and scored lower (0.9561). From 30 s up, the same model was found at about 14 s, so 60 s and 120 s added cost with no gain. A larger or harder dataset would benefit more from a longer budget.

## Likely Tutor Questions

**What is AutoML and why FLAML?** AutoML automates choosing the algorithm and tuning hyperparameters. FLAML is open-source, lightweight and finds good models quickly within a time budget.

**Why build a baseline first?** Without a baseline you cannot tell if AutoML is actually better. It showed me a simple model was just as good here.

**Why did the scaled baseline win?** Logistic regression is sensitive to feature scale. Unscaled, it did not converge. Scaling fixed that, and this dataset is small and nearly linearly separable, so a simple model works well.

**Why is the CV score (0.978) higher than the test score (0.9649)?** CV is measured on training data folds; the test set is unseen. With only 114 test samples, one wrong prediction changes accuracy by about 0.0088.

**What does time\_budget do?** It limits how many seconds FLAML searches. More time means more cost, but here results stopped improving after about 14 s.

**Why set a seed?** So the search and results can be repeated (reproducibility).

**Why the 15.7 s vs 13.3 s difference for the 60 s run?** Timing varies slightly between runs because of the machine's load; the chosen model and score were the same.

**What did a human still decide?** Dataset, split, metric, time budget, a fair baseline, spotting the scaling problem, and whether the model is good enough.

**How does this link to KaizenML?** I measured, found a problem, improved one thing at a time and re-checked. That is continuous improvement (Plan-Do-Check-Act).

**How would you explain the model's predictions?** Use SHAP or permutation feature importance to see which features drive predictions. That is Unit 5.5, and a good next step.

**Why save and reload the model?** So it can be reused or deployed without retraining. Reloading gave the same result, proving it saved correctly.