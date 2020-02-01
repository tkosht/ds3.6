import pandas
import datetime
from matplotlib import pyplot
from fbprophet import Prophet
import utils


class Transer(object):
    def __init__(self):
        pass

    def fit(self, X, y, **params):
        return self

    def transform(self, X, **params):
        return X


class Preprocess(Transer):
    def __init__(self):
        pass

    def fit(self, train_df: pandas.DataFrame, y: pandas.DataFrame, **params):
        return self

    def transform(self, train_predict_df, **params):
        return train_predict_df


class Estimator(object):
    def fit(self, X, y, **params):
        return self

    def predict(self, X, **params):
        return X


class EstimatorProphet(Estimator):
    def __init__(self, holidays_df: pandas.DataFrame, exogs=[]):
        self.model = self.create_prophet_model(holidays_df)
        self.exogs = exogs
        for _exg in exogs:
            self.model.add_regressor(_exg, prior_scale=0.5, mode="multiplicative")
        self.trained_df = None
        self.trained_date = None

    @staticmethod
    def create_prophet_model(holidays_df) -> Prophet:
        return Prophet(
            # mcmc_samples=10,  # for debugging
            mcmc_samples=300,
            holidays=holidays_df,
            holidays_prior_scale=0.25,
            changepoint_prior_scale=0.01,
            seasonality_mode="multiplicative",
            yearly_seasonality=10,
            weekly_seasonality=True,
            daily_seasonality=False,
        )

    def fit(self, X: pandas.DataFrame, y: pandas.DataFrame, **params):
        assert "ds" in X.columns  # date series
        assert "y" in X.columns  # actual values
        assert set(self.exogs).issubset(set(X.columns))
        # y will not be used
        train_df = X.copy()
        self.model.fit(train_df)
        self.trained_df = train_df
        self.trained_date = train_df.ds.iloc[-1]
        return self

    def predict(self, predict_df, **params) -> pandas.DataFrame:
        by_date = self.trained_date
        if "by_date" in params:
            by_date = params["by_date"]
        freq = "D"
        if "freq" in params:
            freq = params.get("freq")
        futures = self.make_futures(predict_df, by_date, freq)

        forecast_df = self.model.predict(futures)
        return forecast_df

    def make_futures(
        self, predict_df: pandas.DataFrame, by_date: str = "2020-12-31", freq="D"
    ):
        by_date = datetime.datetime.strptime(by_date, "%Y-%m-%d")
        predict_df = predict_df.copy()
        future_date = self.model.make_future_dataframe(
            periods=len(predict_df), freq=freq
        )
        trained_date = datetime.datetime.strptime(self.trained_date, "%Y-%m-%d")
        predict_date = trained_date + datetime.timedelta(1)
        future_date = pandas.date_range(predict_date, by_date).to_frame()

        if len(self.exogs) == 0:
            return future_date
        df = pandas.concat([self.trained_df, predict_df], axis=0)
        df.index = df.ds
        data_df = future_date.merge(df, how="left", left_index=True, right_index=True)
        data_df.ds = data_df.index
        data_df = data_df.reset_index(drop=True)
        data_df.reset_index()
        data_df = data_df.interpolate(method="polynomial", order=5)
        data_df = data_df.fillna(0)

        cols = ["ds"]
        cols.extend(self.exogs)  # add regressor columns
        futures = data_df[cols]
        return futures


def make_plot_block(verif, start_date, end_date, ax=None):
    df = verif.loc[start_date:end_date, :]
    df.loc[:, "yhat"].plot(lw=2, ax=ax, color="r", ls="-", label="forecasts")
    ax.fill_between(
        df.index,
        df.loc[:, "yhat_lower"],
        df.loc[:, "yhat_upper"],
        color="coral",
        alpha=0.3,
    )
    df.loc[:, "y"].plot(lw=2, ax=ax, color="steelblue", ls="-", label="observations")
    ax.grid(ls=":")
    ax.legend(fontsize=15)
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]
    ax.set_ylabel("cyclists number", fontsize=15)
    ax.set_xlabel("", fontsize=15)
    ax.set_title(f"{start_date} to {end_date}", fontsize=18)


def save_plot(img_file, forecast_df, train_df, test_df):
    verif = utils.make_verif(forecast_df, train_df, test_df)
    _, axes = pyplot.subplots(nrows=3, figsize=(14, 16), sharey=True)
    ax = axes[0]
    make_plot_block(verif, "2017-01-01", "2017-06-30", ax=ax)
    ax = axes[1]
    make_plot_block(verif, "2017-07-01", "2017-12-31", ax=ax)
    ax = axes[2]
    make_plot_block(verif, "2018-01-01", "2018-06-30", ax=ax)
    ax.set_xlim(["2018-01-01", "2018-06-30"])
    pyplot.savefig(img_file)


if __name__ == "__main__":
    from sklearn.pipeline import make_pipeline

    holidays_df = pandas.read_csv("data/holiday.tsv", sep="\t", header=0)
    data_df = pandas.read_csv("data/data.tsv", sep="\t", header=0)
    train_df = pandas.read_csv("data/train.tsv", sep="\t", header=0)
    test_df = pandas.read_csv("data/test.tsv", sep="\t", header=0)
    assert "ds" in train_df.columns
    assert "y" in train_df.columns
    assert set(train_df) == set(test_df)

    # without exog
    exogs = []
    steps = [
        Preprocess(),
        EstimatorProphet(holidays_df, exogs),
    ]
    pipe = make_pipeline(*steps)

    pipe.fit(train_df, y=None)
    forecast_df = pipe.predict(test_df, by_date="2019-12-31")
    save_plot("img/forecasted_simple.png", forecast_df, train_df, test_df)

    # with exog
    exogs = ["temp", "rain", "sun", "wind"]
    steps = [
        Preprocess(),
        EstimatorProphet(holidays_df, exogs),
    ]
    pipe = make_pipeline(*steps)

    pipe.fit(train_df, y=None)
    forecast_df = pipe.predict(test_df, by_date="2019-12-31")
    save_plot("img/forecasted_exog.png", forecast_df, train_df, test_df)

    print("OK")
