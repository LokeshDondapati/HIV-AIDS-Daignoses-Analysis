import pandas as pd


class DataSummary:
    """
    Class to get data from a URL and showcase data.

    Parameters:
    - url (str): URL of the dataset.
    """

    def __init__(self, url):
        self.url = url

    def read_data(self):
        """
        For loading the dataset, the dataset was uploaded to GitHub, and with the GitHub URL, all the data is being fetched.
        """
        data = pd.read_csv(self.url)
        return data

    def second_read_data(self):
        """
        Reads the second dataset from a predefined URL.

        Returns:
        - pd.DataFrame: Second dataset.
        """
        second_url = "https://raw.githubusercontent.com/LokeshDondapati/HIV-AIDS-Diagnoses-Analysis/main/Datasets/HIV_AIDS_Diagnoses_by_Neighborhood__Sex__and_Race_Ethnicity_20231126.csv"
        data = pd.read_csv(second_url)
        return data

    def data_summary(self):
        """
        Gives rows/use cases and columns/attributes.

        Returns:
        dict: Number of use cases, attributes, data types, and the dataframe.
        """
        # Method that gives data of rows and columns
        data = self.read_data()
        # Calculate the number of use cases
        num_use_cases = data.shape[0]

        # Calculate the number of attributes
        num_attributes = data.shape[1]

        # List data types for each attribute
        data_types = data.dtypes

        # Display the results
        print("Use Cases:", num_use_cases)
        print("Attributes:", num_attributes)
        print("Data Types for Attribute:", data_types)
        result = {
            "Use Cases": num_use_cases,
            "Attributes": num_attributes,
            "Data Types for Attribute": data_types,
            "Dataframe": data,
        }
        return result

    def second_dataset_summary(self):
        """
        Gives rows/use cases and columns/attributes.

        Returns:
        dict: Number of use cases, attributes, and data types.
        """
        # Method that gives data of rows and columns
        data = self.second_read_data()
        # Calculate the number of use cases
        num_use_cases = data.shape[0]

        # Calculate the number of attributes
        num_attributes = data.shape[1]

        # List data types for each attribute
        data_types = data.dtypes

        # Display the results
        print("Use Cases:", num_use_cases)
        print("Attributes:", num_attributes)
        print("Data Types for Attribute:", data_types)
        result = {
            "Use Cases": num_use_cases,
            "Attributes": num_attributes,
            "Data Types for Attribute": data_types,
        }
        return result
