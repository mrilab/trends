import sys
from pytz import timezone
from datetime import datetime
from pandas import DataFrame

from .runner import Runner

if __name__ == "__main__":
    data_dir, log_dir, out_dir = sys.argv[1:]
    basename = datetime.now(tz=timezone("Asia/Taipei")).strftime("%m%d%H%M%S")
    print(f"Session {basename}")
    log_fp = open(f"{log_dir}/{basename}.csv", "w")
    log_fp.write(",fit_time,score_time,test_r2,test_neg_mean_squared_error\n")
    out_fp = open(f"{out_dir}/{basename}.csv", "w")
    out_fp.write("Id,Predicted\n")
    runner = Runner(data_dir)
    for target in runner.dataset.training.y.columns:
        print(f"Training the model for {target}")
        scores, result = runner.run_on_target(target)
        scores = DataFrame(DataFrame.from_dict(scores).mean(), columns=[target]).T
        print(scores)
        scores.to_csv(log_fp, header=False)
        result = DataFrame(result, index=[f"{_}_{target}" for _ in runner.dataset.test.X.index])
        result.to_csv(out_fp, header=False)
    log_fp.close()
    out_fp.close()
