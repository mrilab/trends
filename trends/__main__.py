import sys
from pytz import timezone
from datetime import datetime
from pandas import DataFrame

from .data import get_data
from .model import get_model

if __name__ == "__main__":
    data_dir, log_dir, out_dir = sys.argv[1:]
    dataset = get_data(data_dir)
    basename = datetime.now(tz=timezone("Asia/Taipei")).strftime("%m%d%H%M%S")
    print(f"Session {basename}")
    scores = DataFrame()
    results = DataFrame()
    for target in dataset.training.y.columns:
        model = get_model()
        model.fit(
            dataset.training.X,
            dataset.training.y[target],
        )
        score = DataFrame.from_dict(model.cv_results_)
        scores = scores.append(score)
        print(model.best_params_)
        print(model.best_score_)
        result = model.predict(dataset.test.X)
        result = DataFrame(result, index=[f"{_}_{target}" for _ in dataset.test.X.index])
        results = results.append(result)
    scores.to_csv(f"{log_dir}/{basename}.csv")
    results.index.name = "Id"
    results.columns = ["Predicted"]
    results.to_csv(f"{out_dir}/{basename}.csv")
