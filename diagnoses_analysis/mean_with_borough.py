import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# URL of the dataset
url = "https://raw.githubusercontent.com/LokeshDondapati/project1/main/HIV_AIDS_Diagnoses_by_Neighborhood__Age_Group__and_Race_Ethnicity.csv"


class Inference:
    def __init__(self):
        """Initialize the Inference class with the dataset URL."""
        self.url = url

    def read_data(self):
        """
        Read and preprocess the dataset.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        # Read the dataset
        df = pd.read_csv(self.url)
        return df

    def analysis_seaborn(self):
        """
        Perform Seaborn analysis on the dataset.

        Returns:
            Histogram
        """
        # Read data
        data = self.read_data()

        # Cleaning the 'TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES' columns by replacing '000*' with NaN
        data["TOTAL NUMBER OF HIV DIAGNOSES"] = pd.to_numeric(
            data["TOTAL NUMBER OF HIV DIAGNOSES"].replace("000*", None, regex=True),
            errors="coerce",
        )
        data["TOTAL NUMBER OF AIDS DIAGNOSES"] = pd.to_numeric(
            data["TOTAL NUMBER OF AIDS DIAGNOSES"].replace("000*", None, regex=True),
            errors="coerce",
        )

        # Calculate mean diagnosis rate for borough
        mean_diagnosis_by_borough = (
            data.groupby(["Borough"])[
                ["TOTAL NUMBER OF HIV DIAGNOSES", "TOTAL NUMBER OF AIDS DIAGNOSES"]
            ]
            .mean()
            .reset_index()
        )

        # Melt the dataframe to use "hue" for differentiating between HIV and AIDS diagnoses
        melted_data = pd.melt(
            mean_diagnosis_by_borough,
            id_vars=["Borough"],
            value_vars=[
                "TOTAL NUMBER OF HIV DIAGNOSES",
                "TOTAL NUMBER OF AIDS DIAGNOSES",
            ],
            var_name="Diagnosis Type",
            value_name="Mean of Total Diagnoses",
        )

        # Plot using seaborn
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x="Borough",
            y="Mean of Total Diagnoses",
            hue="Diagnosis Type",
            data=melted_data,
            palette={
                "TOTAL NUMBER OF HIV DIAGNOSES": "blue",
                "TOTAL NUMBER OF AIDS DIAGNOSES": "green",
            },
        )

        plt.title("Mean of HIV/AIDS Diagnoses by boroughs (seaborn)")
        plt.xlabel("Borough")
        plt.ylabel("Mean of Total Diagnoses")
        plt.legend(title="Diagnosis Type", bbox_to_anchor=(1.05, 1), loc="upper right")
        result = plt.show()
        return result

    def analysis_matplotlib(self):
        """
        Perform Matplotlib analysis on the dataset.

        Returns:
            Histogram with subplots
        """
        data = self.read_data()

        # Cleaning the 'TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES' columns by replacing '000*' with NaN
        data["TOTAL NUMBER OF HIV DIAGNOSES"] = pd.to_numeric(
            data["TOTAL NUMBER OF HIV DIAGNOSES"].replace("000*", None, regex=True),
            errors="coerce",
        )
        data["TOTAL NUMBER OF AIDS DIAGNOSES"] = pd.to_numeric(
            data["TOTAL NUMBER OF AIDS DIAGNOSES"].replace("000*", None, regex=True),
            errors="coerce",
        )

        # Calculate mean diagnosis rate for each SEX
        mean_diagnosis_by_sex = (
            data.groupby(["Borough"])[
                ["TOTAL NUMBER OF HIV DIAGNOSES", "TOTAL NUMBER OF AIDS DIAGNOSES"]
            ]
            .mean()
            .reset_index()
        )

        # Melt the dataframe for better visualization
        melted_data = pd.melt(
            mean_diagnosis_by_sex,
            id_vars=["Borough"],
            value_vars=[
                "TOTAL NUMBER OF HIV DIAGNOSES",
                "TOTAL NUMBER OF AIDS DIAGNOSES",
            ],
            var_name="Diagnosis Type",
            value_name="Mean of Total Diagnoses",
        )

        # Plotting using matplotlib
        plt.figure(figsize=(12, 6))

        # Plotting with Matplotlib
        bar_width = 0.35
        bar_positions_hiv = range(len(melted_data) // 2)
        bar_positions_aids = [pos + bar_width for pos in bar_positions_hiv]

        # Plotting HIV diagnoses
        plt.bar(
            bar_positions_hiv,
            melted_data[
                melted_data["Diagnosis Type"] == "TOTAL NUMBER OF HIV DIAGNOSES"
            ]["Mean of Total Diagnoses"],
            width=bar_width,
            label="HIV Diagnoses",
        )

        # Plotting AIDS diagnoses
        plt.bar(
            bar_positions_aids,
            melted_data[
                melted_data["Diagnosis Type"] == "TOTAL NUMBER OF AIDS DIAGNOSES"
            ]["Mean of Total Diagnoses"],
            width=bar_width,
            label="AIDS Diagnoses",
        )

        # Adding labels and title
        plt.title("Mean HIV/AIDS Diagnoses by SEX (Matplotlib)")
        plt.xlabel("Borough")
        plt.ylabel("Mean Total Number of Diagnoses")
        plt.xticks(
            [pos + bar_width / 2 for pos in bar_positions_hiv],
            melted_data["Borough"].unique(),
        )

        # Adding legend
        plt.legend(title="Diagnosis Type")

        result = plt.show()
        return result
