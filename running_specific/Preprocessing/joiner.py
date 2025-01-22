import pandas as pd

# Load the two CSV files
csv1 = "all_runs.csv"  # Replace with the path to your first CSV file
csv2 = "fit_files_summary.csv"  # Replace with the path to your second CSV file

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

# Remove the timezone from the 'start_datetime' column in df2
df2['start_date'] = df2['start_date'].str.replace(r'\+00:00', '', regex=True)

# Perform a full join on 'start_date' and 'start_datetime'
merged_df = pd.merge(df1, df2, how='outer', left_on='start_datetime', right_on='start_date')

# Save the merged DataFrame to a new CSV file
output_csv = "merged_output.csv"
merged_df.to_csv(output_csv, index=False)

print(f"Merged CSV saved as {output_csv}")

