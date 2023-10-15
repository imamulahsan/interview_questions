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

13. What is ensemble method?\
Answer: Ensemble methods in machine learning are techniques that involve combining the predictions from multiple base models (individual machine learning models) to create a more robust, accurate, and reliable model. eg. Bagging (Bootstrap Aggregating), Boosting, Stacking.

# Unsupervised Learning

14. What is unspervised learning?\
Answer: Unsupervised learning is a category of machine learning where the algorithm is trained on data without explicit supervision or labeled output. In unsupervised learning, the model's objective is to discover hidden patterns, structures, or relationships within the data.

15. What is dimension reduction?\
Answer: Dimension reduction in machine learning refers to the process of reducing the number of input variables or features in a dataset while retaining as much relevant information as possible. It is primarily used to simplify the data, improve computational efficiency, and mitigate the "curse of dimensionality," which can lead to overfitting and increased model complexity.

16. What is Principal Component Analysis?\
Answer: Principal Component Analysis (PCA) is a dimensionality reduction technique and a popular unsupervised learning method in machine learning and statistics. PCA is used to transform high-dimensional data into a lower-dimensional representation while retaining as much of the original variance or information as possible.

17. What is SVD?\
Answer: Singular Value Decomposition (SVD) is a mathematical technique used in machine learning and various data analysis tasks. It is a matrix factorization method that decomposes a given matrix into three simpler matrices, allowing data to be represented in a lower-dimensional form. SVD is widely used for dimensionality reduction, data compression, noise reduction, and feature extraction in machine learning and other fields.

18. Applications of clustering.\
Answer: Customer Segmentation, Image Segmentation, Anomaly Detection, Document Clustering.

19. K-means clustering.
Answer: K-Means clustering is one of the most widely used and straightforward clustering algorithms in machine learning and data analysis. It is an unsupervised learning technique that partitions data into clusters, where each data point belongs to the cluster with the nearest mean. K-Means is based on the concept of minimizing the sum of squared distances between data points and their respective cluster centroids.

20. How to choose the number of cluster in machine learning?\
Answer: hoosing the number of clusters in a clustering algorithm is an important decision, and it can significantly impact the quality and interpretability of your results. There is no one-size-fits-all method for determining the optimal number of clusters, but there are several techniques and heuristics that can help guide your decision e.g. Elbow Method, Silhouette Score.

# Model Selection and Evaluation

21. Differentiate Overfitting and Underfitting.\
Answer: Overfitting occurs when a model learns to fit the training data too closely, capturing noise and random fluctuations rather than the true underlying patterns. Such a model will perform poorly on new data because it has essentially memorized the training data. In contrast, underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance both on the training and new data.

22. Differentiate Bias-Variance Trade-off.\
Answer: Achieving good generalization involves finding a balance between bias and variance. High bias (underfitting) means the model is too simple and makes strong assumptions about the data. High variance (overfitting) means the model is too complex and fits the training data too closely. An optimal model generalizes well by finding the right level of complexity that captures the true patterns in the data without fitting the noise.

23. What is decision tree?\
Answer: A decision tree is a popular and interpretable machine learning model used for both classification and regression tasks. It is a tree-like structure that recursively divides the dataset into subsets based on the values of input features, ultimately making a decision or prediction at the leaf nodes of the tree.

24. What is random forest?\
Answer: Random Forest is an ensemble machine learning technique that is widely used for classification and regression tasks. It is a versatile and powerful algorithm that leverages the idea of creating multiple decision trees and combining their predictions to improve accuracy, reduce overfitting, and enhance model robustness.

