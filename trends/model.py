from numpy import (arange, linspace, logspace)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.svm import (SVR, NuSVR)
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV

def get_model():
    return GridSearchCV(
        estimator=make_pipeline(
            StandardScaler(),
            PCA(whiten=True),
            ElasticNet(),
        ),
        param_grid={
            "pca__n_components": arange(0.5, 1, 0.05),
            "elasticnet__alpha": logspace(-5, 5, 11, base=4),
            "elasticnet__l1_ratio": linspace(0, 1, 11),
        },
        scoring="r2",
        refit=True,
        cv=5,
        verbose=1,
        n_jobs=-1,
    )
