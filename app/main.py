import yaml
import pandas
from fbprophet import Prophet
import utility.plotter as plotter


def load_params():
    with open("main.yml", "r") as f:
        params = yaml.load(f)
    return params


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
    from sklearn.pipeline import make_pipeline
    from model.prophet import PreprocessProphet, EstimatorProphet
    from dataset.auckset import DatasetCyclicAuckland

    params = load_params()

    freq = params["prediction"]["freq"]
    predict_date = params["prediction"]["predict_date"]
    predict_by = params["prediction"]["predict_by"]

    holidays_df = pandas.read_csv("data/holiday.tsv", sep="\t", header=0)
    dcaset = DatasetCyclicAuckland(freq=freq)

    # split dataset
    train_df, test_df = dcaset.split(predict_date)

    predict_params = dict(predict_by=predict_by, freq=freq)

    # without exog
    m = create_model(holidays_df)
    exogs = []
    steps = [
        PreprocessProphet(),
        EstimatorProphet(holidays_df, exogs, m),
    ]
    pipe = make_pipeline(*steps)

    pipe.fit(train_df, y=None)
    forecast_df = pipe.predict(test_df, **predict_params)
    forecast_df.to_csv("data/forecasted_simple.tsv", sep="\t", header=True, index=False)
    plotter.save_plot("img/forecasted_simple.png", forecast_df, train_df, test_df)

    # with exog
    m = create_model(holidays_df)
    exogs = ["temp", "rain", "sun", "wind"]
    steps = [
        PreprocessProphet(),
        EstimatorProphet(holidays_df, exogs, m),
    ]
    pipe = make_pipeline(*steps)

    pipe.fit(train_df, y=None)
    forecast_df = pipe.predict(test_df, **predict_params)
    forecast_df.to_csv("data/forecasted_exog.tsv", sep="\t", header=True, index=False)
    plotter.save_plot("img/forecasted_exog.png", forecast_df, train_df, test_df)

    print("OK")
