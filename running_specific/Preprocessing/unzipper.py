
from fit2gpx import Converter

conv = Converter()


import os
import pandas as pd
import warnings

# Suppress all warnings globally
warnings.filterwarnings("ignore")

def process_fit_files(folder_path, output_csv):
    # Gather all .fit files in the folder
    fit_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.endswith(".fit")
    ]

    total_files = len(fit_files)  # Total number of files
    results = []  # List to store results

    # Process each file
    for idx, file_path in enumerate(fit_files, start=1):
        print(f"Processing file {idx} of {total_files}: {os.path.basename(file_path)}")

        try:
            # Process the file to get dataframes
            df_lap, df_point = conv.fit_to_dataframes(fname=file_path)

            # Calculate metrics
            start_date = df_point['timestamp'].min()
            average_hr = df_point["heart_rate"].mean().round(1)
            max_hr = df_point["heart_rate"].max().round(1)

            # Add the results to the list
            results.append({
                "start_date": start_date,
                "average_heart_rate": average_hr,
                "max_heart_rate": max_hr
            })

        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Define the folder containing the .fit files and the output CSV file path
folder_path = "../bulk_strava/activities"
output_csv = "fit_files_summary.csv"

# Process the files and save the results
process_fit_files(folder_path, output_csv)
