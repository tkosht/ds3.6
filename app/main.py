import datetime
import pathlib
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot
from fbprophet import Prophet
import utility.plotter as plotter


class DatasetCyclicAuckland(object):
    def __init__(self, data_file="data/data.tsv"):
        self.data_df = None
        self.data_file = data_file
        self._load()
        self.train_df = None
        self.test_df = None

    def _load(self):
        data_df = pandas.read_csv(self.data_file, sep="\t", header=0, parse_dates=True)
        data_df.rename(columns=dict(datetime="ds"), inplace=True)
        data_df.index = data_df.ds
        assert "ds" in data_df.columns
        assert "y" in data_df.columns
        self.data_df = data_df
        return self

    def split(self, predict_date: str):
        train_df = self.data_df[:predict_date].iloc[:-1]
        test_df = self.data_df[predict_date:]
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        assert "ds" in train_df.columns
        assert "y" in train_df.columns
        assert "ds" in test_df.columns
        assert "y" in test_df.columns
        return train_df, test_df


def save_plot(img_file, forecast_df, train_df, test_df, plot_freq="6M"):
    verif = plotter.make_verif(forecast_df, train_df, test_df)
    s = verif.ds.iloc[0]
    e = verif.ds.iloc[-1]
    intervals = pandas.date_range(s, e, freq=plot_freq) + datetime.timedelta()
    if intervals[-1] < e:
        date_end = intervals[-1] + relativedelta(months=6)
        intervals = pandas.Series(intervals).append(pandas.Series(date_end))

    n = len(intervals) - 1
    _, axes = pyplot.subplots(nrows=n, figsize=(14, 16), sharey=True)

    for idx, (s, e) in enumerate(zip(intervals[:-1], intervals[1:])):
        s = s.strftime("%Y/%m/%d")
        e = e.strftime("%Y/%m/%d")
        ax = axes[idx]
        plotter.make_plot_block(verif, s, e, ax=ax)
    pathlib.Path("img").mkdir(parents=True, exist_ok=True)
    pyplot.savefig(img_file)


def create_model(holidays_df) -> Prophet:
    return Prophet(
        # mcmc_samples=10,  # for debugging
        mcmc_samples=100,
        holidays=holidays_df,
        holidays_prior_scale=0.25,
        changepoint_prior_scale=0.01,
        # seasonality_mode="multiplicative",
        seasonality_mode="additive",
        yearly_seasonality=10,
        weekly_seasonality=True,
        daily_seasonality=False,
    )


if __name__ == "__main__":
    import pandas
    from sklearn.pipeline import make_pipeline
    from model.prophet import Preprocess, EstimatorProphet

    predict_date = "2017-01-01"
    predict_by = "2019-12-31"

    holidays_df = pandas.read_csv("data/holiday.tsv", sep="\t", header=0)
    dcaset = DatasetCyclicAuckland()

    # split dataset
    train_df, test_df = dcaset.split(predict_date)

    # without exog
    m = create_model(holidays_df)
    exogs = []
    steps = [
        Preprocess(),
        EstimatorProphet(holidays_df, exogs, m),
    ]
    pipe = make_pipeline(*steps)

    pipe.fit(train_df, y=None)
    forecast_df = pipe.predict(test_df, by_date=predict_by)
    save_plot("img/forecasted_simple.png", forecast_df, train_df, test_df)

    # with exog
    m = create_model(holidays_df)
    exogs = ["temp", "rain", "sun", "wind"]
    steps = [
        Preprocess(),
        EstimatorProphet(holidays_df, exogs, m),
    ]
    pipe = make_pipeline(*steps)

    pipe.fit(train_df, y=None)
    forecast_df = pipe.predict(test_df, by_date=predict_by)
    save_plot("img/forecasted_exog.png", forecast_df, train_df, test_df)

    print("OK")
