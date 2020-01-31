import numpy
import pandas
from matplotlib import pyplot
import seaborn as sns
from fbprophet import Prophet
import utils

def create_prophet_model(holidays_df):
    return Prophet(mcmc_samples=300, \
            holidays=holidays_df, \
            holidays_prior_scale=0.25, \
            changepoint_prior_scale=0.01, \
            seasonality_mode='multiplicative', \
            yearly_seasonality=10, \
            weekly_seasonality=True, \
            daily_seasonality=False)


def make_futures(model, data_df, exogs=[], sep_date="2017"):
    d = data_df.copy()
    d.index = pandas.to_datetime(d.datetime)
    test_df = d[sep_date:]
    test_df = test_df.reset_index(drop=True)
    test_df.rename(columns=dict(datetime="ds"), inplace=True)
    future = model.make_future_dataframe(periods=len(test_df), freq='1D')
    if len(exogs) == 0:
        return future

    def _merge(future, data_df):
        regressors = data_df[exogs]
        regressors.index = pandas.to_datetime(data_df.datetime)
        futures = future.copy() 
        futures.index = pandas.to_datetime(futures.ds)
        futures = futures.merge(regressors, left_index=True, right_index=True)
        futures = futures.reset_index(drop=True)
        return futures

    return _merge(future, data_df)


if __name__ == "__main__":
    holidays_df = pandas.read_csv("data/holiday.tsv", sep="\t", header=0)
    data_df = pandas.read_csv("data/data.tsv", sep="\t", header=0)
    train_df = pandas.read_csv("data/train.tsv", sep="\t", header=0)
    test_df = pandas.read_csv("data/test.tsv", sep="\t", header=0)

    model = create_prophet_model(holidays_df)

    exogs = ["temp", "rain", "sun", "wind"]
    for _exg in exogs:
        model.add_regressor(_exg, prior_scale=0.5, mode='multiplicative')
    model.fit(train_df)

    futures = make_futures(model, data_df, exogs)
    print(futures.head(3))

    print("hello")
