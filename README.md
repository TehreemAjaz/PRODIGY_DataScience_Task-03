# PRODIGY_DataScience_Task-03

Code Description:
This Python script builds a decision tree classifier using the 'Bank Marketing' dataset from the UCI Machine Learning Repository. The dataset contains various demographic and behavioral features of customers, such as age, job type, marital status, and previous marketing campaign results. The classifier predicts whether a customer will subscribe to a term deposit (the target variable y).

Step-by-Step Explanation:
Import Necessary Libraries: The script begins by importing essential libraries like pandas for data manipulation, zipfile and requests for downloading and extracting the dataset, and scikit-learn libraries (train_test_split, DecisionTreeClassifier, accuracy_score, classification_report, confusion_matrix, and plot_tree) for building and evaluating the decision tree model.

Download and Extract the Dataset: The dataset is hosted in a ZIP file on the UCI Machine Learning Repository. The code uses the requests library to download the ZIP file and zipfile to extract its contents. The CSV file within the ZIP is then loaded into a pandas DataFrame.

Display Initial Columns: To help identify the target variable, the script prints the column names of the DataFrame.

Preprocess the Data: The target column y (indicating whether a customer subscribed to the product) is isolated. The remaining features are one-hot encoded using pd.get_dummies() to convert categorical variables into a binary format suitable for machine learning.

Split the Data into Training and Testing Sets: The data is split into training and testing sets using train_test_split, with 70% of the data used for training and 30% for testing. This is done to evaluate the model's performance on unseen data.

Initialize and Train the Decision Tree Classifier: The decision tree model is initialized using DecisionTreeClassifier and then trained on the training data (X_train and y_train).

Make Predictions and Evaluate the Model: The trained model makes predictions on the test set (X_test). The script evaluates the model using several metrics:

Accuracy: The proportion of correctly predicted instances over the total instances.
Classification Report: Provides detailed metrics like precision, recall, F1-score, and support for each class.
Confusion Matrix: Shows the count of true positive, true negative, false positive, and false negative predictions, providing insight into the model's performance.
Visualize the Decision Tree: Finally, the script plots the decision tree using plot_tree from scikit-learn to visualize the decision-making process of the classifier. This visualization helps understand how the model uses the features to make predictions.
