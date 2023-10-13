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

4. What is data transformation in machine learning?\
Answer: Data transformation in machine learning refers to the process of altering the format, structure, or values of data to make it more suitable for a particular machine learning algorithm or task.

5. If the distribution of test data (not yet seen by the model) is significantly different than the distribution of the training data, what problems can occur? What transformations can be applied to data before passing them to an ML model and why should these transformations be performed?\
Answer: If the test data has a different distribution than the training data used to build a model it will likely cause poor performance.

6. What standardization and normalization?\
Answer: Standardization transforms the data into a standard normal distribution with a mean (average) of 0 and a standard deviation of 1. Normalization, also known as Min-Max scaling, scales the data to a specific range, typically between 0 and 1.

# Supervised Learning

7. What is supervised learning?\
Answer: Supervised learning is a type of machine learning where an algorithm learns from labeled training data to make predictions or decisions without being explicitly programmed.

8. What is overfitting?\
Answer: Overfitting is a common problem in machine learning, especially in supervised learning, where a model learns the training data too well, to the point that it starts capturing noise or random fluctuations in the data rather than the underlying patterns. When a model overfits, it performs exceptionally well on the training data but poorly on unseen or test data.

9. 
