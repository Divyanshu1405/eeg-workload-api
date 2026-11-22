"""Picklable wrapper around XGBClassifier + LabelEncoder for production serving."""

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBWrapper:
    def __init__(self, clf: XGBClassifier, label_encoder: LabelEncoder):
        self.clf = clf
        self.le = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X):
        preds_enc = self.clf.predict(X)
        return self.le.inverse_transform(preds_enc)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
