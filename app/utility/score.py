import numpy
import pandas


class Score(object):
    def __init__(self, a: numpy.ndarray, p: numpy.ndarray):
        self.a = numpy.array(a)  # actual/observation values
        self.p = numpy.array(p)  # predicted/forecast values
        assert len(self.a) == len(self.p)

    def mae(self):
        return numpy.abs((self.a - self.p)).mean()

    def mse(self):
        return ((self.a - self.p) ** 2).mean()

    def rmse(self):
        return self.mse() ** 0.5

    def mape(self):
        a = self._adjusted_values(self.a)
        return numpy.abs(((a - self.p) / self.a)).mean()

    def mspe(self):
        a = self._adjusted_values(self.a)
        return (((a - self.p) / self.a) ** 2).mean()

    def rmspe(self):
        return self.mspe() ** 0.5

    def _adjusted_values(self, v):
        v = v.copy()
        v[v == 0] = v[v != 0].mean()
        v[v == 0] = 1.0  # if case of all 0 constant values
        return v

    def sqmrpa(self):  # square measurement sqmr of prediction/actual
        a = self._adjusted_values(self.a)
        return self.p.sum() / a.sum()

    def sqmrap(self):  # square measurement sqmr of actual/prediction
        p = self._adjusted_values(self.p)
        return self.a.sum() / p.sum()

    def gmpa(self):
        a = self._adjusted_values(self.a)
        r = self.p / a
        return r.prod() ** (1 / len(r))

    def gmap(self):
        p = self._adjusted_values(self.p)
        r = self.a / p
        return r.prod() ** (1 / len(r))

    def rsq(self):  # R square
        srs = ((self.a - self.p) ** 2).sum()  # sum of residuals squared
        srm = ((self.a - self.a.mean()) ** 2).sum()  # sum of residual from mean
        return 1 - srs / srm

    @property
    def df(self) -> pandas.DataFrame:
        s = dict(
            mae=self.mae(),
            mse=self.mse(),
            rmse=self.rmse(),
            mape=self.mape(),
            mspe=self.mspe(),
            rmspe=self.rmspe(),
            sqmrpa=self.sqmrpa(),
            sqmrap=self.sqmrap(),
            gmpa=self.gmpa(),
            gmap=self.gmap(),
            rsq=self.rsq(),
        )
        df = pandas.DataFrame(s, index=["scores"])
        return df.T

    def to_csv(self, tsv_file, sep=","):
        self.df.to_csv(tsv_file, sep=sep, header=False)

    def to_json(self, json_file):
        self.df.to_json(json_file)

    @property
    def json(self) -> str:
        return self.df.to_json()


if __name__ == "__main__":
    y = numpy.array([1.1, 1.2, 1.3])
    p = numpy.array([1.0, 0.9, 1.4])
    score = Score(y, p)
    print(score.df)
    print(score.json)
    score.to_tsv("score.tsv")
    score.to_json("score.json")
