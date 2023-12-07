import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset URL
url = "https://raw.githubusercontent.com/LokeshDondapati/project1/main/HIV_AIDS_Diagnoses_by_Neighborhood__Age_Group__and_Race_Ethnicity.csv"


class EdaAnalysis:
    """
    Class to get data from URL and showcase data.
    Show Summary of statistics stored in a DataFrame called "summary_stats."

    Parameters:
    - url (str): URL of the dataset.
    """

    def __init__(self):
        self.url = url

    def read_data(self, url):
        """For loading dataset, the dataset was uploaded to GitHub, and with the GitHub URL, all the data is being fetched"""
        df = pd.read_csv(url)
        # Extract start and end ages from the 'AGE' column
        df[["start_age", "end_age"]] = df["AGE"].str.split("-", expand=True)

        # Convert age columns to numeric, handling non-numeric values
        df["start_age"] = pd.to_numeric(
            df["start_age"].str.replace(r"\D+", "", regex=True),
            errors="coerce",
            downcast="integer",
        )
        df["end_age"] = pd.to_numeric(
            df["end_age"].str.replace(r"\D+", "", regex=True),
            errors="coerce",
            downcast="integer",
        )

        # Fill missing values in 'end_age' with a default value (60)
        df["end_age"].fillna(60, inplace=True)

        # For rows with 'All', set the start_age to 0 and end_age to a large number (e.g., 100)
        df.loc[df["AGE"] == "All", ["start_age", "end_age"]] = [0, 100]

        # Create a combined age column representing the age range
        df["combined_age"] = df.apply(
            lambda row: f"{int(row['start_age'])}+"
            if row["end_age"] == 60
            else f"{int(row['start_age'])} - {int(row['end_age'])}",
            axis=1,
        )

        return df

    def second_read_data(self):
        """
        Reads the second dataset from a predefined URL.

        Returns:
        - pd.DataFrame: Second dataset.
        """
        second_url = "https://raw.githubusercontent.com/LokeshDondapati/HIV-AIDS-Daignoses-Analysis/main/Datasets/HIV_AIDS_Diagnoses_by_Neighborhood__Sex__and_Race_Ethnicity_20231126.csv"
        data = pd.read_csv(second_url)
        return data

    def statistics(self):
        """Summary statistics for the DataFrame are calculated with describe func."""
        # Method read_data is called for data fetch
        summary_stats = self.read_data(self.url).describe(include="all")
        return summary_stats

    def gender_dataset_statistics(self):
        """Summary statistics for the DataFrame are calculated with describe func."""
        # Method second_read_data is called for data fetch
        summary_stats = self.second_read_data().describe(include="all")
        return summary_stats

    def graphical_analysis_matplotlib_year(self):
        """
        Dimensions/size are given 8, 6, and color as sky blue for Matplotlib histogram.

        Graph axis:
        - X: Year.
        - Y: Frequency.

        Returns:
        Histogram of year using Matplotlib.
        """
        plt.figure(figsize=(8, 6))
        # Method read_data is used here for year
        plt.hist(
            self.read_data(self.url)["YEAR"],
            bins=20,
            color="skyblue",
            edgecolor="black",
            width=0.8,  # Adjust the width based on your preference
        )
        plt.xlabel("YEAR")
        plt.ylabel("Frequency")
        plt.title("Histogram for YEAR (Matplotlib)")
        result = plt.show()
        return result

    def graphical_analysis_matplotlib_age(self):
        """
        Dimensions/size are given 10, 8, and color as sky blue for Matplotlib histogram.

        Graph axis:
        - X: Age.
        - Y: Frequency.

        Returns:
        Histogram of age.
        """
        plt.figure(figsize=(10, 8))  # Adjust the size as needed

        plt.hist(
            self.read_data(self.url)["combined_age"],
            bins=10,
            color="blue",
            edgecolor="black",
            rwidth=0.8,
        )
        plt.xlabel("Combined Age")
        plt.ylabel("Frequency")
        plt.title("Histogram for AGE")
        result = plt.show()

        return result

    def graphical_analysis_seaborn_year(self):
        """
        Dimensions/size are given 8, 6, and color as sky blue for Seaborn histogram.

        Graph axis:
        - X: Year.
        - Y: Frequency.

        Returns:
        Histogram of year.
        """
        plt.figure(figsize=(8, 6))

        sns.histplot(self.read_data(self.url)["YEAR"], bins=30, color="blue", kde=True)
        plt.xlabel("YEAR")
        plt.ylabel("Frequency")
        plt.title("Histogram for YEAR")
        result = plt.show()
        return result

    def graphical_analysis_seaborn_age(self):
        """
        Dimensions/size are given 8, 6, and color as sky blue for Seaborn histogram.

        Graph axis:
        - X: Age.
        - Y: Frequency.

        Returns:
        Histogram of age.
        """
        sns.histplot(
            self.read_data(self.url)["combined_age"], bins=50, color="blue", kde=True
        )
        plt.xlabel("Combined Age")
        plt.ylabel("Frequency")
        plt.title("Histogram for AGE")
        result = plt.show()
        return result
