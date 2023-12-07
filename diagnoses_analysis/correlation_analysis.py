import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class CorrelationAnalysis:
    """
    Class for analyzing correlations in a DataFrame and visualizing the correlation matrix.
    """

    def __init__(self, dataframe):
        """
        Initialize the CorrelationAnalysis class with a DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame for correlation analysis.
        """
        self.dataframe = dataframe

    def plot_correlation_heatmap(self):
        """
        Plot a heatmap of the correlation matrix for the DataFrame.

        Returns:
        None
        """
        correlation_matrix = self.dataframe.corr()

        # Plot the correlation matrix heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix of Encoded Features")
        plt.show()

    def display_highest_correlations(self):
        """
        Display the highest correlations in the correlation matrix in ascending order.

        Returns:
        None
        """
        # Assuming 'correlation_matrix' is your correlation matrix
        corr_series = self.dataframe.corr().unstack().sort_values(ascending=False)

        # Exclude self-correlations and duplicates
        corr_series = corr_series[corr_series != 1.0]

        # Display highest correlations in ascending order
        highest_correlations = corr_series.head()
        print(highest_correlations)


