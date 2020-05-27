from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.svm import (SVR, NuSVR)
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV

def get_model():
    pipeline = make_pipeline(
        PCA(whiten=False),
        Lasso(),
    )
    param_grid = {
        "pca__n_components": [8, 16, 32, 64, 128, 256, 512],
        "lasso__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }
    metrics = "r2"
    searcher = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=metrics,
        refit=True,
        cv=5,
        verbose=1,
        n_jobs=-1,
    )
    return searcher
