import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = 'scores'

# Function to calculate the score for a row
def calculate_score(row):
    # Initialize the count of classes with non-zero scores
    non_zero_classes = 9
    # Iterate over the last 36 columns (9 classes * 4 metrics)
    for i in range(4, 40, 4):
        # Check if all 4 metrics for the class are zero
        if all(row[i:i+4] == 0):
            non_zero_classes -= 1
    # Calculate the score
    return (non_zero_classes / 9) * 100

# Dictionary to store results for each file
results = {}

# Iterate over all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # Load the CSV file
        data = pd.read_csv(file_path)
        # Calculate scores for each row
        scores = data.apply(calculate_score, axis=1)
        # Store the scores in the results dictionary
        results[file_name] = scores

# Print the results
for file_name, scores in results.items():
    print(f"Scores for {file_name}:")
    print(scores)
    print()