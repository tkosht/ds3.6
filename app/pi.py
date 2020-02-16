import yaml
import numpy
import pandas
import joblib
from utility.score import Score


def load_params() -> dict:
    with open("conf/main.yml", "r") as f:
        params = yaml.full_load(f)
    return params


class PermutationImportance(object):
    def __init__(self, model, predict_params: dict):
        self.model = model
        self.predict_params = predict_params
        self.scores = {}
        self.score_keys = []

    def make_scores(self, df: pandas.DataFrame):
        self._calculate(df)._arrange()

    def _calculate(self, df: pandas.DataFrame):
        fdf = self.model.predict(df, **self.predict_params)
        scr0 = Score(df.y, fdf.yhat.iloc[: len(df)])
        self.features = df.drop(columns=["ds", "y"]).columns
        scores = {}
        for idx, ftr in enumerate(self.features):
            _df = df.copy()
            sr = _df[ftr]
            psr = sr.take(numpy.random.permutation(len(sr)))
            psr.reset_index(drop=True, inplace=True)
            _df = _df.assign(**{ftr: psr})
            _fdf = self.model.predict(_df, **self.predict_params)
            _scr = Score(_df.y, _fdf.yhat.iloc[: len(_df)])
            scores[ftr] = _scr.df / scr0.df
        self.scores = scores
        self.score_keys = sorted(scr0.dic.keys())
        return self

    def _arrange(self):
        pi_scores = {ky: {} for ky in self.score_keys}
        for ky in pi_scores.keys():
            for ftr in self.features:
                pi_scores[ky][ftr] = self.scores[ftr].loc[ky].values[0]
        self.pi_scores = pi_scores
        return self

    def do_print(self):
        for ky in self.score_keys:
            pi = sorted(self.pi_scores[ky].items(), key=lambda x: x[1], reverse=True)
            print(ky, pi)

    @property
    def df(self) -> pandas.DataFrame:
        score_df = pandas.DataFrame([])
        for ky in self.score_keys:
            df = pandas.DataFrame(self.pi_scores[ky], index=[ky])
            score_df = pandas.concat([score_df, df], axis=0)
        return score_df

    def to_csv(self, filename, sep="\t"):
        self.df.to_csv(filename, sep=sep)


def load_model(dump_file):
    return joblib.load(dump_file)


if __name__ == "__main__":
    from dataset.auckset import DatasetCyclicAuckland

    params = load_params()

    freq = params["prediction"]["freq"]
    predict_date = params["prediction"]["predict_date"]
    predict_by = params["prediction"]["predict_by"]

    holidays_df = pandas.read_csv("data/holiday.tsv", sep="\t", header=0)
    dcaset = DatasetCyclicAuckland(freq=freq)

    # split dataset
    train_df, test_df = dcaset.split(predict_date)

    # setup params
    fit_params = {"model__" + k: v for k, v in params["model"]["fit"].items()}
    predict_params = dict(predict_by=predict_by, freq=freq)

    pipe = load_model("data/model/pipe.exog.logistic.gz")
    pi = PermutationImportance(pipe, predict_params)
    pi.make_scores(df=test_df)
    pi.do_print()
    print(pi.df)

    print("OK")
