import kagglehub

# Downloads to: kagglehub/datasets/wordsforthewise/lending-club/versions/3/
# Accepted loans CSV: accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv
# Rejected loans CSV: rejected_2007_to_2018q4.csv/rejected_2007_to_2018Q4.csv
path = kagglehub.dataset_download("wordsforthewise/lending-club")

print("Path to dataset files:", path)
