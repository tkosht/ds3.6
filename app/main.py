import yaml
import pandas

from fbprophet import Prophet
from typing import Union
import utility.plotter as plotter


def load_params() -> dict:
    with open("conf/main.yml", "r") as f:
        params = yaml.load(f)
    return params


def create_model(
    holidays_df, **params
) -> Union[
    Prophet,
]:
    model_name = list(params["model"].keys())[0]
    model_params = params["model"][model_name]
    model_class = eval(model_name)
    if model_class == Prophet:
        model_params.update(dict(holidays=holidays_df))
    return model_class(**model_params)


if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from model.prophet import PreprocessProphet, EstimatorProphet
    from dataset.auckset import DatasetCyclicAuckland
    from utility.score import Score

    params = load_params()

    freq = params["prediction"]["freq"]
    predict_date = params["prediction"]["predict_date"]
    predict_by = params["prediction"]["predict_by"]

    holidays_df = pandas.read_csv("data/holiday.tsv", sep="\t", header=0)
    dcaset = DatasetCyclicAuckland(freq=freq)

    # split dataset
    train_df, test_df = dcaset.split(predict_date)

    fit_params = dict(model__thin=2, model__chains=5, model__seed=777)
    predict_params = dict(predict_by=predict_by, freq=freq)

    # without exog
    m = create_model(holidays_df, **params)
    exogs = []
    steps = [
        ("preprocess", PreprocessProphet()),
        ("model", EstimatorProphet(holidays_df, exogs, m)),
    ]
    pipe = Pipeline(steps=steps)

    pipe.fit(train_df, y=None, **fit_params)
    forecast_df = pipe.predict(test_df, **predict_params)
    forecast_df.to_csv("data/forecasted_simple.tsv", sep="\t", header=True, index=False)
    plotter.save_plot(
        f"img/forecasted_simple_{m.growth}.png", forecast_df, train_df, test_df
    )
    plotter.save_plot_components(
        f"img/components_simple_{m.growth}.png",
        m,
        forecast_df.drop(columns=["weekly"]),
    )

    scr = Score(test_df.y, forecast_df.yhat.iloc[: len(test_df)])
    scr.to_csv(f"data/score_simple_{m.growth}.tsv", sep="\t")

    # with exog
    m = create_model(holidays_df, **params)
    exogs = params["data"]["exogs"]
    steps = [
        ("preprocess", PreprocessProphet()),
        ("model", EstimatorProphet(holidays_df, exogs, m)),
    ]
    pipe = Pipeline(steps=steps)

    pipe.fit(train_df, y=None, **fit_params)
    forecast_df = pipe.predict(test_df, **predict_params)
    forecast_df.to_csv("data/forecasted_exog.tsv", sep="\t", header=True, index=False)
    plotter.save_plot(
        f"img/forecasted_exog_{m.growth}.png", forecast_df, train_df, test_df
    )
    plotter.save_plot_components(
        f"img/components_exog_{m.growth}.png", m, forecast_df.drop(columns=["weekly"]),
    )
    scr = Score(test_df.y, forecast_df.yhat.iloc[: len(test_df)])
    scr.to_csv(f"data/score_exog_{m.growth}.tsv", sep="\t")

    print("OK")
