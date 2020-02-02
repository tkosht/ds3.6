import pandas
import datetime
from fbprophet import Prophet


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
    def __init__(self, holidays_df: pandas.DataFrame, exogs=[], model: Prophet = None):
        self.model = model
        if model is None:
            self.model = self.create_prophet_model(holidays_df)
        self.exogs = exogs
        for _exg in exogs:
            self.model.add_regressor(_exg, prior_scale=0.5, mode="multiplicative")
        self.trained_df = None
        self.trained_date = None

    @staticmethod
    def create_prophet_model(holidays_df) -> Prophet:
        return Prophet(
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
        trained_date = datetime.datetime.strptime(self.trained_date, "%Y-%m-%d")
        predict_date = trained_date + datetime.timedelta(1)
        future_date = (
            pandas.date_range(predict_date, by_date, freq=freq, name="ds")
            .to_frame()
            .reset_index(drop=True)
        )

        if len(self.exogs) == 0:
            return future_date

        df = pandas.concat([self.trained_df, predict_df], axis=0)
        df.ds = pandas.to_datetime(df.ds)
        data_df = future_date.merge(df, how="left", on="ds")
        data_df.reset_index(drop=True, inplace=True)
        data_df = data_df.interpolate(method="polynomial", order=5)
        data_df = data_df.fillna(0)  # fill far futures

        cols = ["ds"]
        cols.extend(self.exogs)  # add regressor columns
        futures = data_df[cols]
        return futures
