from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import (SVR, NuSVR)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_validate

from .data import *

class Runner:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dataset = get_data(data_dir)
        self.estimator = make_pipeline(
            PCA(n_components=32, whiten=False),
            NuSVR(),
        )
        self.metrics = ["r2", "neg_mean_squared_error"]

    def get_targets(self):
        return self.dataset.training.y.columns

    def run_on_target(self, target):
        scores = cross_validate(
            self.estimator,
            self.dataset.training.X,
            self.dataset.training.y[target],
            scoring=self.metrics,
            cv=5, n_jobs=5,
        )
        self.estimator.fit(
            self.dataset.training.X,
            self.dataset.training.y[target],
        )
        result = self.estimator.predict(self.dataset.test.X)
        return scores, result
