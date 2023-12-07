import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Dataset URL
url = "https://raw.githubusercontent.com/LokeshDondapati/project1/main/HIV_AIDS_Diagnoses_by_Neighborhood__Age_Group__and_Race_Ethnicity.csv"


class Inference:
    def __init__(self):
        """
        Constructor to initialize the Inference class.
        """
        self.url = url

    def readdata(self, url):
        """
        Reads and processes the data from the provided URL.

        Parameters:
        - url (str): The URL of the dataset.

        Returns:
        - pd.DataFrame: Processed DataFrame.
        """
        # Read the CSV file into a DataFrame
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

    def readdata_second(self):
        """
        Reads the second dataset from a predefined URL.

        Returns:
        - pd.DataFrame: Second dataset.
        """
        file_path = "https://raw.githubusercontent.com/LokeshDondapati/HIV-AIDS-Daignoses-Analysis/main/Datasets/HIV_AIDS_Diagnoses_by_Neighborhood__Sex__and_Race_Ethnicity_20231126.csv"
        data = pd.read_csv(file_path)
        return data

    def analysis_seaborn(self):
        """
        Performs analysis using Seaborn library and visualizes the results.
        """
        # Load data using the readdata method
        data = self.readdata(url)

        # Convert a specific column to numeric, handling special characters
        data["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"] = pd.to_numeric(
            data["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"].replace(
                "000*", None, regex=True
            ),
            errors="coerce",
        )

        # Filter rows based on age conditions
        data_filtered = data[(data["start_age"] != 0) | (data["end_age"] != 100)]

        # Group data for visualization
        grouped_data = (
            data_filtered.groupby(["combined_age", "RACE/ETHNICITY"])[
                "TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"
            ]
            .sum()
            .reset_index()
        )

        # Define a custom color palette for the plot
        custom_palette = {
            "Asian/Pacific Islander": "orange",
            "Black": "green",
            "Latino/Hispanic": "red",
            "White": "blue",
            "Hispanic": "brown",
            "Other/Unknown": "purple",
            "Unknown": "yellow",
            "Native American": "grey",
            "Multiracial": "skyblue",
            "All": "black",
            "Asian/Pacific\nIslander": "orange",
        }

        # Create a subplot for Seaborn plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        bar_width = 0.2

        # Plot the first analysis on the first subplot using Seaborn
        sns.barplot(
            x="combined_age",
            y="TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES",
            hue="RACE/ETHNICITY",
            data=grouped_data,
            palette=custom_palette,
            ax=axes[0],
            dodge=True,
            ci=None,
        )

        axes[0].set_title("Seaborn plot")
        axes[0].set_ylabel("Sum of TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES")
        axes[0].set_xlabel("Combined Age")
        axes[0].tick_params(axis="x", rotation=90)

        df = self.readdata_second()

        # Convert a specific column to numeric, handling special characters
        df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"] = pd.to_numeric(
            df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"], errors="coerce"
        )  # Convert to numeric

        # Handle non-finite values and convert to integers
        df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"] = (
            df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"].fillna(0).astype(int)
        )

        # Filter the DataFrame to exclude rows with 'All' in 'RACE/ETHNICITY' or 'SEX'
        filtered_df = df[(df["RACE/ETHNICITY"] != "All") & (df["SEX"] != "All")]

        # Group by both 'RACE/ETHNICITY' and 'SEX' and sum the 'TOTAL NUMBER OF HIV DIAGNOSES' column
        sum_per_ethnicity_sex_filtered = filtered_df.groupby(["RACE/ETHNICITY", "SEX"])[
            "TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"
        ].sum()

        # Prepare data for Matplotlib plotting
        grouped_data = (
            filtered_df.groupby(["RACE/ETHNICITY", "SEX"])[
                "TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"
            ]
            .sum()
            .unstack()
        )

        # Sort values within each 'RACE/ETHNICITY' category in ascending order
        grouped_data = grouped_data.apply(
            lambda x: x.sort_values(ascending=True), axis=1
        )

        # Define the order of 'RACE/ETHNICITY' based on the Matplotlib plot
        ethnicity_order = (
            grouped_data.index.tolist()
        )  # Use the sorted order from Matplotlib

        # Define a custom color palette to match Matplotlib colors
        custom_palette = {
            "Female": "orange",
            "Male": "blue",
        }  # Replace with your desired colors

        # Define the hue order for the 'SEX' categories and reverse it
        hue_order = ["Female", "Male"]  # Reverse the order

        # Seaborn Plot with synchronized colors and reversed hue order
        sns.barplot(
            x="RACE/ETHNICITY",
            y="TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES",
            hue="SEX",
            data=filtered_df,
            estimator=sum,
            ci=None,
            order=ethnicity_order,
            palette=custom_palette,
            hue_order=hue_order,
        )
        plt.title("Seaborn plot")
        plt.xlabel("Ethnicity")
        plt.ylabel("Sum of TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES")
        plt.xticks(rotation=45)
        plt.legend(title="Sex", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Display the plot
        plt.show()

    def analysis_matplotlib(self):
        """
        Performs analysis using Matplotlib library and visualizes the results.
        """
        # Load data using the readdata method
        data = self.readdata(url)

        # Clean the 'TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES' column
        data["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"] = pd.to_numeric(
            data["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"].replace(
                "000*", None, regex=True
            ),
            errors="coerce",
        )

        # Filter rows based on age conditions
        data1 = data[(data["start_age"] != 0) | (data["end_age"] != 100)]

        # Group by the specified columns and aggregate using the sum
        grouped_data_filtered = (
            data1.groupby(["combined_age", "NEIGHBORHOOD", "RACE/ETHNICITY"])[
                "TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"
            ]
            .sum()
            .reset_index()
        )

        # Custom palette for visualization
        custom_palette = {
            "Asian/Pacific Islander": "orange",
            "Black": "green",
            "Latino/Hispanic": "red",
            "White": "blue",
            "Hispanic": "brown",
            "Other/Unknown": "purple",
            "Unknown": "yellow",
            "Native American": "grey",
            "Multiracial": "skyblue",
            "All": "black",
        }

        # Plotting the bar plot using Matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Matplotlib plot with separate bars for each race/ethnicity
        age_groups = np.unique(grouped_data_filtered["combined_age"])
        # Adjust the width based on your preference
        bar_width = 0.15

        for i, (race, color) in enumerate(custom_palette.items()):
            # Filter data for the specific race/ethnicity
            race_data = grouped_data_filtered[
                grouped_data_filtered["RACE/ETHNICITY"] == race
            ]

            # Ensure that the age groups are consistent
            ages = np.unique(race_data["combined_age"])

            # Find the indices for plotting
            indices = np.arange(len(ages)) + i * bar_width

            # Find counts for the specific race/ethnicity
            counts = race_data.groupby("combined_age")[
                "TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"
            ].sum()

            # Plot the bar
            axes[0].bar(indices, counts, width=bar_width, color=color, label=race)

        axes[0].set_title("Matplotlib")
        axes[0].set_ylabel("Sum of TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES")
        axes[0].set_xlabel("Combined Age")
        axes[0].set_xticks(
            np.arange(len(ages)) + (len(custom_palette) - 1) * bar_width / 2
        )
        axes[0].set_xticklabels(ages)
        axes[0].tick_params(axis="x", rotation=90)

        # Add legend to the Matplotlib plot
        axes[0].legend(
            title="RACE/ETHNICITY",
            bbox_to_anchor=(1.05, 1),
            loc="upper right",
            fontsize="small",
        )

        df = self.readdata_second()

        df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"] = pd.to_numeric(
            df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"], errors="coerce"
        )  # Convert to numeric

        # Handle non-finite values and convert to integers
        df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"] = (
            df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"].fillna(0).astype(int)
        )

        # Filtering the DataFrame to exclude rows with 'All' in 'RACE/ETHNICITY' or 'SEX'
        filtered_df = df[(df["RACE/ETHNICITY"] != "All") & (df["SEX"] != "All")]

        # Grouping by both 'RACE/ETHNICITY' and 'SEX' and summing the 'TOTAL NUMBER OF HIV DIAGNOSES' column
        sum_per_ethnicity_sex_filtered = filtered_df.groupby(["RACE/ETHNICITY", "SEX"])[
            "TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"
        ].sum()

        # Prepare data for Matplotlib plotting
        grouped_data = (
            filtered_df.groupby(["RACE/ETHNICITY", "SEX"])[
                "TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"
            ]
            .sum()
            .unstack()
        )

        # Sort values within each 'RACE/ETHNICITY' category in ascending order
        grouped_data = grouped_data.apply(
            lambda x: x.sort_values(ascending=True), axis=1
        )

        # Matplotlib Plot with reversed hue
        bar_width = 0.35
        index = range(len(grouped_data.index))

        # Reverse the iteration order to reverse the hue
        for i, sex in reversed(list(enumerate(grouped_data.columns))):
            axes[1].bar(
                [x + (i * bar_width) for x in index],
                grouped_data[sex],
                width=bar_width,
                label=sex,
            )

        axes[1].set_xlabel("Ethnicity")
        axes[1].set_ylabel("Sum of TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES")
        axes[1].set_title("Matplotlib")
        axes[1].set_xticks([x + bar_width for x in index])
        axes[1].set_xticklabels(grouped_data.index)
        axes[1].legend(title="Sex", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Display the plot
        plt.show()
