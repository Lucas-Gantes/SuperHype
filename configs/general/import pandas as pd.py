import pandas as pd

class EntitySet:
    def __init__(self, dataframe):
        self._dataframe = dataframe

    def group_elements(self, col1, col2):
        elements = self._dataframe.groupby(col1, observed=False)[col2].unique().to_dict()
        return elements