import numpy as np
import pandas as pd
import holidays
import glob
import utils


if __name__ == "__main__":
    lfiles = glob.glob("data/cycling_Auckland/cycling_counts_????.csv")
    lfiles.sort()
    df_list = []
    for f in lfiles:
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        df_list.append(d)
    df = pd.concat(df_list, axis=0)
    df = df.loc[:, ["Tamaki Drive EB", "Tamaki Drive WB"]]
    Tamaki = df.loc[:, "Tamaki Drive WB"] + df.loc[:, "Tamaki Drive EB"]
    Tamaki = Tamaki.loc[
        "2013":"2018-06-01",
    ]
    Tamaki = Tamaki.to_frame(name="Tamaki Drive")
    dfc = Tamaki.copy()
    dfc.loc[:, "Tamaki Drive, Filtered"] = utils.median_filter(
        dfc, varname="Tamaki Drive"
    )
    data = dfc.loc["2013":, ["Tamaki Drive, Filtered"]].resample("1D").sum()

    holidays_df = pd.DataFrame([], columns=["ds", "holiday"])
    ldates = []
    lnames = []
    for date, name in sorted(
        holidays.NZ(prov="AUK", years=np.arange(2013, 2018 + 1)).items()
    ):
        ldates.append(date)
        lnames.append(name)
    ldates = np.array(ldates)
    lnames = np.array(lnames)
    holidays_df.loc[:, "ds"] = ldates
    holidays_df.loc[:, "holiday"] = lnames
    holidays_df.loc[:, "holiday"] = holidays_df.loc[:, "holiday"].apply(
        lambda x: x.replace(" (Observed)", "")
    )
    holidays_df.holiday.unique()
    data = data.rename({"Tamaki Drive, Filtered": "y"}, axis=1)
    data_train, data_test = utils.prepare_data(data, 2017)

    # weather data
    temp = pd.read_csv(
        "./data/weather/hourly/commute/temp_day.csv", index_col=0, parse_dates=True
    )
    temp = temp.loc[:, ["Tmin(C)"]]
    rain = pd.read_csv(
        "./data/weather/hourly/commute/rain_day.csv", index_col=0, parse_dates=True
    )
    rain = rain.loc[:, ["Amount(mm)"]]
    sun = pd.read_csv(
        "./data/weather/hourly/commute/sun_day.csv", index_col=0, parse_dates=True
    )
    wind = pd.read_csv(
        "./data/weather/hourly/commute/wind_day.csv", index_col=0, parse_dates=True
    )
    wind = wind.loc[:, ["Speed(m/s)"]]

    temp.columns = ["temp"]
    rain.columns = ["rain"]
    sun.columns = ["sun"]
    wind.columns = ["wind"]

    def _preprocess(df):
        df = df.loc["2013":"2018-06-01", :]
        df = df.interpolate(method="linear")
        return df

    temp = _preprocess(temp)
    rain = _preprocess(rain)
    sun = _preprocess(sun)
    wind = _preprocess(wind)

    data_with_regressors = utils.add_regressor(data, temp, varname="temp")
    data_with_regressors = utils.add_regressor(
        data_with_regressors, rain, varname="rain"
    )
    data_with_regressors = utils.add_regressor(data_with_regressors, sun, varname="sun")
    data_with_regressors = utils.add_regressor(
        data_with_regressors, wind, varname="wind"
    )

    data_train, data_test = utils.prepare_data(data_with_regressors, 2017)

    data_with_regressors.to_csv("data/data.tsv", sep="\t", header=True, index=True)
    data_train.to_csv("data/train.tsv", sep="\t", header=True, index=None)
    data_test.to_csv("data/test.tsv", sep="\t", header=True, index=None)
    holidays_df.to_csv("data/holiday.tsv", sep="\t", header=True, index=None)
