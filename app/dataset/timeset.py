class TimeSeriesDataset(object):
    def __init__(self, data_file="data/data.tsv"):
        self.data_df = None
        self.data_file = data_file

    def _load(self):
        raise NotImplementedError()

    def split(self, predict_date: str):
        raise NotImplementedError()
