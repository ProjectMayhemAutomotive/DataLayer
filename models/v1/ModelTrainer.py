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
# Model Class
class ModelTrainer:
    def setBestModelIndex(self, bestModelIndex: dict):
        self.bestModelIndex = bestModelIndex
    def train_model(self, train_data, target: dict):
        all_extra_cols = target['classification_type']+target['regression_type']
        features = train_data.drop(columns=all_extra_cols)
        classificationLabels = train_data[target['classification_type']]

        # Iterate through each target and train a separate model
        self.models = {}
        for t in target['classification_type']:
            # Use RandomForestClassifier, XGBoost, LightGBM, LogisticRegression for training
            from sklearn.ensemble import RandomForestClassifier
            rfModel = RandomForestClassifier(n_estimators=100, random_state=42)
            xgbModel = xgb.XGBClassifier()
            lightGBMModel = lgb.LGBMClassifier()
            lrModel = LogisticRegression()
            svcModel = SVC()
            # Train the models
            rfModel.fit(features, classificationLabels[t])
            xgbModel.fit(features, classificationLabels[t])
            lightGBMModel.fit(features, classificationLabels[t])
            lrModel.fit(features, classificationLabels[t])
            svcModel.fit(features, classificationLabels[t])
            self.models[t] = [rfModel, xgbModel, lightGBMModel, lrModel, svcModel]
        return self.models
    def evaluate_model(self, test_data, target: list[str], dump_results=False, c: float=2):
        features = test_data.drop(columns=target+['failure_year', 'failure_month', 'failure_day'])
        labels = test_data[target]

        evaluation_results = {}
        for t in target:
            rfModel, xgbModel, lightGBMModel, lrModel, svcModel = self.models[t]
            predictions1 = rfModel.predict(features)
            predictions2 = xgbModel.predict(features)
            predictions3 = lightGBMModel.predict(features)
            predictions4 = lrModel.predict(features)
            predictions5 = svcModel.predict(features)

            evaluation_results[t] = {
                "RandomForest": {
                    "accuracy": accuracy_score(labels[t], predictions1),
                    "precision": precision_score(labels[t], predictions1),
                    "recall": recall_score(labels[t], predictions1),
                    "f1_score": f1_score(labels[t], predictions1)
                },
                "XGBoost": {
                    "accuracy": accuracy_score(labels[t], predictions2),
                    "precision": precision_score(labels[t], predictions2),
                    "recall": recall_score(labels[t], predictions2),
                    "f1_score": f1_score(labels[t], predictions2)
                },
                "LightGBM": {
                    "accuracy": accuracy_score(labels[t], predictions3),
                    "precision": precision_score(labels[t], predictions3),
                    "recall": recall_score(labels[t], predictions3),
                    "f1_score": f1_score(labels[t], predictions3)
                },
                "LogisticRegression": {
                    "accuracy": accuracy_score(labels[t], predictions4),
                    "precision": precision_score(labels[t], predictions4),
                    "recall": recall_score(labels[t], predictions4),
                    "f1_score": f1_score(labels[t], predictions4)
                },
                "SVC": {
                    "accuracy": accuracy_score(labels[t], predictions5),
                    "precision": precision_score(labels[t], predictions5),
                    "recall": recall_score(labels[t], predictions5),
                    "f1_score": f1_score(labels[t], predictions5)
                }
            }

            # Plot Model Evaluation Results
            for model_name, metrics in evaluation_results[t].items():
                print(f"Model: {model_name}")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
                plt.figure(figsize=(6,4))
                cm = confusion_matrix(labels[t],
                                      rfModel.predict(features) if model_name == 'RandomForest' else
                                      xgbModel.predict(features) if model_name == 'XGBoost' else
                                      lightGBMModel.predict(features) if model_name == 'LightGBM' else
                                      lrModel.predict(features) if model_name == 'LogisticRegression' else
                                      svcModel.predict(features))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix for {model_name} on {t}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.show()
                sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
                plt.title(f'Metrics for {model_name} on {t}')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                plt.show()
        if dump_results:
            with open('D:\\EY Hackathon\\Data Layer\\models\\v1\\results.log', 'a') as f:
                jsonStr = "@ c = "+str(c)+"\n" + json.dumps(evaluation_results, indent=4)
                # Go to last line
                f.write(jsonStr)
        return evaluation_results
    def save(self, path: str):
        # Using Pickele to save the model
        with open(path, 'wb') as f:
            pkl.dump(self, f)
    def infer(self, new_data, target: str):
        features = new_data.drop(columns=[target])
        bestModel = self.models[target][self.bestModelIndex[target]]
        prediction = bestModel.predict(features)
        return prediction