import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset URL
url = "https://raw.githubusercontent.com/LokeshDondapati/project1/main/HIV_AIDS_Diagnoses_by_Neighborhood__Age_Group__and_Race_Ethnicity.csv"


class Inference:
    def __init__(self):
        """
        Constructor to initialize the Inference class.
        """
        self.url = url

    def read_data(self):
        """
        Reads and processes the data from the provided URL.

        Returns:
        - pd.DataFrame: Processed DataFrame.
        """
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.url)

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

    def analysis_seaborn(self):
        """
        Analyzes and visualizes the mean of HIV/AIDS diagnoses by age using Seaborn.

        Returns:
        - plt.figure: Seaborn plot showing the mean of diagnoses by age.
        """
        # Read data using the read_data method
        data = self.read_data()

        # Clean the 'TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES' column by replacing '000*' with NaN
        data["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"] = pd.to_numeric(
            data["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"].replace(
                "000*", None, regex=True
            ),
            errors="coerce",
        )

        # Filter rows based on age conditions
        data1 = data[(data["start_age"] != 0) | (data["end_age"] != 100)]

        # Calculate mean diagnosis rate for each age
        mean_diagnosis_by_age = (
            data1.groupby("combined_age")[
                "TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"
            ]
            .mean()
            .reset_index()
        )

        # Plot using Seaborn
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x="combined_age",
            y="TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES",
            data=mean_diagnosis_by_age,
            palette="viridis",
        )
        plt.title("Mean of HIV/AIDS Diagnoses by AGE (seaborn)")
        plt.xlabel("AGE")
        plt.ylabel("Mean of TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES")

        # Display the plot
        result = plt.show()
        return result

    def analysis_matplotlib(self):
        """
        Analyzes and visualizes the mean of HIV/AIDS diagnoses by age using Matplotlib.

        Returns:
        - plt.figure: Matplotlib plot showing the mean of diagnoses by age.
        """
        # Read data using the read_data method
        data = self.read_data()

        # Clean the 'TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES' column by replacing '000*' with NaN
        data["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"] = pd.to_numeric(
            data["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"].replace(
                "000*", None, regex=True
            ),
            errors="coerce",
        )

        # Filter rows based on age conditions
        data1 = data[(data["start_age"] != 0) | (data["end_age"] != 100)]

        # Calculate mean diagnosis rate for each age
        mean_diagnosis_by_age = (
            data1.groupby("combined_age")[
                "TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"
            ]
            .mean()
            .reset_index()
        )

        # Plot using Matplotlib
        plt.figure(figsize=(12, 6))
        plt.bar(
            mean_diagnosis_by_age["combined_age"],
            mean_diagnosis_by_age["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"],
            color="green",
        )
        plt.title("Mean HIV/AIDS Diagnoses by AGE (matplotlib)")
        plt.xlabel("combined_age")
        plt.ylabel("Mean TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES")

        # Display the plot
        result = plt.show()
        return result
