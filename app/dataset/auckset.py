import pandas
from .timeset import TimeSeriesDataset


class DatasetCyclicAuckland(TimeSeriesDataset):
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

    def split(self, predict_date: str) -> (pandas.DataFrame, pandas.DataFrame):
        train_df = self.data_df[:predict_date].iloc[:-1]
        test_df = self.data_df[predict_date:]
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        assert "ds" in train_df.columns
        assert "y" in train_df.columns
        assert "ds" in test_df.columns
        assert "y" in test_df.columns
        return train_df, test_df
