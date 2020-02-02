import pandas


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
