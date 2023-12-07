from diagnoses_analysis import data_summary
from diagnoses_analysis import eda
from diagnoses_analysis import comparison_analysis
from diagnoses_analysis import analysis_mean
from diagnoses_analysis import mean_with_borough
from diagnoses_analysis.diagnoses_prediction import DiagnosisPrediction
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class CodeProcessor:
    """
    A class for processing and analyzing HIV/AIDS diagnosis data.
    """

    def load_dataset(self, url):
        """
        Load dataset from the given URL.

        Args:
        - url: URL of the dataset

        Returns:
        - DataFrame: Loaded dataset
        """
        df = data_summary.DataSummary(url).data_summary()
        return df

    def perform_eda(self, df):
        """
        Perform exploratory data analysis (EDA) on the dataset.

        Args:
        - df: DataFrame to analyze

        Returns:
        - None
        """
        stat = eda.EdaAnalysis().statistics()
        eda.EdaAnalysis().gender_dataset_statistics()
        eda.EdaAnalysis().graphical_analysis_matplotlib_year()
        eda.EdaAnalysis().graphical_analysis_seaborn_year()
        eda.EdaAnalysis().graphical_analysis_matplotlib_age()
        eda.EdaAnalysis().graphical_analysis_seaborn_age()
        comparison_analysis.Inference().analysis_seaborn()
        comparison_analysis.Inference().analysis_matplotlib()
        analysis_mean.Inference().analysis_seaborn()
        analysis_mean.Inference().analysis_matplotlib()
        mean_with_borough.Inference().analysis_seaborn()
        mean_with_borough.Inference().analysis_matplotlib()

    def plot_pair_plots(self, df):
        """
        Plot pair plots to visualize relationships between numerical columns.

        Args:
        - df: DataFrame to visualize

        Returns:
        - None
        """
        numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
        sns.pairplot(df[numerical_columns])
        plt.show()

    def preprocess_data(self, df):
        """
        Preprocess the dataset by handling missing values and encoding categorical columns.

        Args:
        - df: DataFrame to preprocess

        Returns:
        - DataFrame: Preprocessed dataset
        """
        df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"] = (
            pd.to_numeric(
                df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"], errors="coerce"
            )
            .fillna(0)
            .astype(float)
        )

        df_prediction_set = pd.get_dummies(df, columns=["RACE/ETHNICITY"])
        return df_prediction_set

    def visualize_analysis(self):
        """
        Visualize correlation matrix and decision trees.

        Returns:
        - None
        """
        correlation_matrix = df_prediction_set.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix of Encoded Features")
        plt.show()

        DiagnosisPrediction().visualize_trees()

    def train_model(self):
        """
        Train a model and print the Mean Squared Error (MSE).

        Returns:
        - None
        """
        df = analysis_pred.read_data(url)
        df = DiagnosisPrediction().drop_columns(df)
        df = DiagnosisPrediction().convert_to_numeric(df)
        df_prediction_set = self.preprocess_data(df)
        mse = DiagnosisPrediction().train_model(df_prediction_set)
        print(f"Mean Squared Error: {mse}")


# Usage of the CodeProcessor class
processor = CodeProcessor()
url = "https://raw.githubusercontent.com/LokeshDondapati/project1/main/HIV_AIDS_Diagnoses_by_Neighborhood__Age_Group__and_Race_Ethnicity.csv"
df = processor.load_dataset(url)
processor.perform_eda(df)
processor.plot_pair_plots(df)
processor.visualize_analysis()
processor.train_model()
