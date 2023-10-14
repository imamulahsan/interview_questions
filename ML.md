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

9. What is feature selection?\
Answer: Feature selection in machine learning is the process of selecting a subset of the most relevant features (variables or attributes) from a larger set of features. The objective is to improve the model's performance by focusing on the most informative and important features while reducing dimensionality, increasing model interpretability, and potentially speeding up training and inference.

10. Give some common methods for feature selection.
Answer: Filter Methods, Wrapper Methods, Embedded Methods, Principal Component Analysis (PCA).

11. What is Regression?\
Answer: Regression in machine learning is a supervised learning technique used for predicting a continuous numeric value or a real number based on one or more input features.

12. Types of Regression.\
Answer:\
--Linear Regression: This is one of the simplest regression models. It fits a linear relationship between the input features and the target variable, aiming to find the best-fitting straight line through the data.\
--Polynomial Regression: In cases where a linear relationship doesn't adequately capture the data, polynomial regression fits a polynomial function to the data, allowing for more complex relationships.\
--Ridge and Lasso Regression: These are variants of linear regression that introduce regularization terms to prevent overfitting and improve model generalization.\
--Support Vector Regression (SVR): SVR extends the concept of support vector machines to regression tasks and aims to find a hyperplane that best fits the data within a specified margin.\
--Decision Tree Regression: Decision trees can be used for regression by recursively splitting the data into subsets based on the input features and assigning an output value to each leaf node.\
--Random Forest Regression: Random Forest is an ensemble learning technique that combines multiple decision trees to improve prediction accuracy and reduce overfitting in regression tasks.\
--Gradient Boosting Regression: Gradient boosting algorithms, such as Gradient Boosting Trees and XGBoost, build an ensemble of decision trees to make accurate predictions for regression tasks.\
