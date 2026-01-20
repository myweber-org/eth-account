
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape

    def handle_missing_values(self, method='mean', columns=None):
        if columns is None:
            columns = self.df.columns

        for col in columns:
            if self.df[col].isnull().any():
                if method == 'mean':
                    fill_value = self.df[col].mean()
                elif method == 'median':
                    fill_value = self.df[col].median()
                elif method == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif method == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    continue
                else:
                    fill_value = method

                self.df[col] = self.df[col].fillna(fill_value)

        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

        return self

    def standardize_columns(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std > 0:
                self.df[col] = (self.df[col] - mean) / std

        return self

    def get_cleaned_data(self):
        return self.df

    def get_cleaning_report(self):
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]

        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'removed_rows': removed_rows,
            'removed_columns': removed_cols,
            'missing_values_remaining': self.df.isnull().sum().sum()
        }

        return report

def clean_dataset(df, missing_method='mean', remove_outliers=True):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(method=missing_method)

    if remove_outliers:
        cleaner.remove_outliers_iqr()

    cleaner.standardize_columns()
    return cleaner.get_cleaned_data(), cleaner.get_cleaning_report()