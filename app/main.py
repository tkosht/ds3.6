import yaml
import pandas

from fbprophet import Prophet
from typing import Union
import utility.plotter as plotter


def load_params() -> dict:
    with open("conf/main.yml", "r") as f:
        params = yaml.full_load(f)
    return params


def create_model(
    holidays_df, **params
) -> Union[
    Prophet,
]:
    model_name = params["model"]["class"]
    model_params = params["model"]["init"]
    model_class = eval(model_name)
    if model_class == Prophet:
        model_params.update(dict(holidays=holidays_df))
    return model_class(**model_params)


def dump_model(model, dump_file):
    pathlib.Path(dump_file).parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, dump_file, compress=("gzip", 3))


if __name__ == "__main__":
    import joblib
    import pathlib
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

    # setup params
    fit_params = {"model__" + k: v for k, v in params["model"]["fit"].items()}
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
    dump_model(pipe, f"data/model/pipe.simple.{m.growth}.gz")

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
    dump_model(pipe, f"data/model/pipe.exog.{m.growth}.gz")

    print("OK")
