import pandas as pd

csv_path = "~/paparazzi/test_folder/test_images/AE4317_2019_datasets/cyberzoo_poles/20190121-135121.csv"
df = pd.read_csv(csv_path)

print(df.head())  # Print first few rows
print(df.columns)  # See column names
