import pickle

from readnext.config import ResultsPaths

with open(ResultsPaths.evaluation.feature_weights_candidates_pkl, "rb") as f:
    feature_weights_candidates = pickle.load(f)
