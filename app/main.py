import pathlib
import pandas
from matplotlib import pyplot
from fbprophet import Prophet


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
    verif = make_verif(forecast_df, train_df, test_df)
    _, axes = pyplot.subplots(nrows=3, figsize=(14, 16), sharey=True)
    ax = axes[0]
    make_plot_block(verif, "2017-01-01", "2017-06-30", ax=ax)
    ax = axes[1]
    make_plot_block(verif, "2017-07-01", "2017-12-31", ax=ax)
    ax = axes[2]
    make_plot_block(verif, "2018-01-01", "2018-06-30", ax=ax)
    ax.set_xlim(["2018-01-01", "2018-06-30"])
    pathlib.Path("img").mkdir(parents=True, exist_ok=True)
    pyplot.savefig(img_file)


def make_verif(forecast, data_train, data_test):
    forecast = forecast.copy()
    data_train = data_train.copy()
    data_test = data_test.copy()

    forecast.index = pandas.to_datetime(forecast.ds)
    data_train.index = pandas.to_datetime(data_train.ds)
    data_test.index = pandas.to_datetime(data_test.ds)
    data = pandas.concat([data_train, data_test], axis=0)
    forecast.loc[:, "y"] = data.loc[:, "y"]
    return forecast


def create_model(holidays_df) -> Prophet:
    return Prophet(
        # mcmc_samples=10,  # for debugging
        mcmc_samples=100,
        holidays=holidays_df,
        holidays_prior_scale=0.25,
        changepoint_prior_scale=0.01,
        seasonality_mode="multiplicative",
        yearly_seasonality=10,
        weekly_seasonality=True,
        daily_seasonality=False,
    )


if __name__ == "__main__":
    from sklearn.pipeline import make_pipeline
    from model.prophet import Preprocess, EstimatorProphet

    holidays_df = pandas.read_csv("data/holiday.tsv", sep="\t", header=0)
    data_df = pandas.read_csv("data/data.tsv", sep="\t", header=0)
    train_df = pandas.read_csv("data/train.tsv", sep="\t", header=0)
    test_df = pandas.read_csv("data/test.tsv", sep="\t", header=0)
    assert "ds" in train_df.columns
    assert "y" in train_df.columns
    assert set(train_df) == set(test_df)

    # without exog
    m = create_model(holidays_df)
    exogs = []
    steps = [
        Preprocess(),
        EstimatorProphet(holidays_df, exogs, m),
    ]
    pipe = make_pipeline(*steps)

    pipe.fit(train_df, y=None)
    forecast_df = pipe.predict(test_df, by_date="2019-12-31")
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
    forecast_df = pipe.predict(test_df, by_date="2019-12-31")
    save_plot("img/forecasted_exog.png", forecast_df, train_df, test_df)

    print("OK")
