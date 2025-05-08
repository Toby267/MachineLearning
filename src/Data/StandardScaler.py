import pandas as pd

class StandardScaler:
    means: dict[str: float]
    variances: dict[str: float]
    
    def __init__(self):
        pass

    def fitTransform(self, df: pd.DataFrame) -> pd.DataFrame:
        """fits the transform for the given data frame and applies it to said data frame"""

        self.means = {key: df[key].mean() for key in df.columns}
        self.variances = {key: df[key].std() for key in df.columns}
        
        for seriesName, series in df.items():
            for i, value in series.items():
                df.at[i, seriesName] = (value - self.means[seriesName]) / self.variances[seriesName]

        return df
