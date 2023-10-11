# Data Pre-processing and Visualization
---
1. What is a general Machine Learning pipeline?\
Answer:  Import --> Instantiate(Preprocess and Object call) --> Fit (Hyperparameter tuning + Train/Test Split) --> Predict --> Evaluate

2. How to handle missing data?\
Answer: Omission(Remove rows and columns) and Imputation (Fill with zero, mean, median, mode)

3. Python functions for handling missing values\
Answer:

|  Function  | Description |
| --------  | -------- |
| df.isna().sum()      | No. of zeroes in the dataframe |
| df['feature'].mean()  | Find mean |
| .shape      | Dimension |
| df.columns     | See the name of the columns |
| .fillna(0)      | Fill missing values with zereos |
