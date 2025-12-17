# Import Required packages
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle as pkl
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
RESULT_LOG_PATH = r"D:\EY Hackathon\Data Layer\models\v1\results.log"
# Model Class
class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model_index = {}

    def set_best_model_index(self, best_model_index: dict):
        self.best_model_index = best_model_index

    def train_model(self, train_data, target: dict):
        all_targets = target['classification_type'] + target['regression_type']
        X = train_data.drop(columns=all_targets)

        # classification
        for target_col in target['classification_type']:
            y = train_data[target_col]

            models = [
                RandomForestClassifier(n_estimators=100, random_state=42),
                xgb.XGBClassifier(verbosity=0, random_state=42),
                lgb.LGBMClassifier(random_state=42),
                LogisticRegression(max_iter=1000, random_state=42),
                SVC(random_state=42)
            ]

            for model in models:
                model.fit(X, y)

            self.models[target_col] = models

        # regression
        for target_col in target['regression_type']:
            y = train_data[target_col]

            models = [
                RandomForestRegressor(n_estimators=100, random_state=42),
                xgb.XGBRegressor(verbosity=0, random_state=42),
                lgb.LGBMRegressor(random_state=42),
                LinearRegression()
            ]

            for model in models:
                model.fit(X, y)

            self.models[target_col] = models

    # evaluation
    def evaluate_model(self, test_data, target: dict, dump_results=False, c=2):
        all_targets = target['classification_type'] + target['regression_type']
        X = test_data.drop(columns=all_targets)
        y_all = test_data[all_targets]

        results = {}

        # classification
        clf_names = ['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression', 'SVC']

        for target_col in target['classification_type']:
            results[target_col] = {}
            y_true = y_all[target_col]

            for model_name, model in zip(clf_names, self.models[target_col]):
                y_pred = model.predict(X)

                metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1_score': f1_score(y_true, y_pred, zero_division=0)
                }

                results[target_col][model_name] = metrics
                self._plot_classification(y_true, y_pred, metrics, model_name, target_col)

        # regression
        reg_names = ['RandomForest', 'XGBoost', 'LightGBM', 'LinearRegression']

        for target_col in target['regression_type']:
            results[target_col] = {}
            y_true = y_all[target_col]

            for model_name, model in zip(reg_names, self.models[target_col]):
                y_pred = model.predict(X)

                metrics = {
                    'MSE': np.mean((y_true - y_pred) ** 2),
                    'MAE': np.mean(np.abs(y_true - y_pred)),
                    'R_Squared': r2_score(y_true, y_pred)
                }

                results[target_col][model_name] = metrics
                self._plot_regression(y_true, y_pred, metrics, model_name, target_col)

        if dump_results:
            self._save_results(results, c)

        return results

    # plotting
    def _plot_classification(self, y_true, y_pred, metrics, model, target):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{model} - {target}")

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
        ax[0].set_title("Confusion Matrix")

        ax[1].bar(metrics.keys(), metrics.values())
        ax[1].set_ylim(0, 1.05)
        ax[1].set_title("Metrics")

        plt.tight_layout()
        plt.show()

    def _plot_regression(self, y_true, y_pred, metrics, model, target):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{model} - {target}")

        ax[0].scatter(y_true, y_pred, alpha=0.6)
        ax[0].plot([y_true.min(), y_true.max()],
                   [y_true.min(), y_true.max()], 'r--')
        ax[0].set_title("Actual vs Predicted")

        ax[1].bar(metrics.keys(), metrics.values())
        ax[1].set_title("Metrics")

        plt.tight_layout()
        plt.show()

    # save/infer methods
    def _save_results(self, results, c):
        with open(RESULT_LOG_PATH, 'a') as f:
            f.write(f"@ c = {c}\n")
            f.write(json.dumps(results, indent=4))
            f.write("\n\n")

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    def infer(self, new_data, target):
        model = self.models[target][self.best_model_index[target]]
        return model.predict(new_data)