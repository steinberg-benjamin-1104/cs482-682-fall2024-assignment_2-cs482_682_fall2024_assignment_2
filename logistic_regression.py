import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

class MyLogisticRegression:
    def __init__(self, dataset_num, perform_test):
        self.training_set = None
        self.test_set = None
        self.model_logistic = None
        self.model_linear = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.perform_test = perform_test
        self.dataset_num = dataset_num
        self.read_csv(self.dataset_num)

    def read_csv(self, dataset_num):
        if dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        elif dataset_num == '2':
            train_dataset_file = 'train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        else:
            raise ValueError("Unsupported dataset number")
            
        self.training_set = pd.read_csv(train_dataset_file, sep=',', header=0)
        self.X_train = self.training_set[['exam_score_1', 'exam_score_2']]
        self.y_train = self.training_set['label']
        
        if self.perform_test:
            self.test_set = pd.read_csv(test_dataset_file, sep=',', header=0)
            self.X_test = self.test_set[['exam_score_1', 'exam_score_2']]
            self.y_test = self.test_set['label']
        
    def model_fit_linear(self):
        '''Initialize and fit the linear regression model.'''
        self.model_linear = LinearRegression()
        self.model_linear.fit(self.X_train, self.y_train)
    
    def model_fit_logistic(self):
        '''Initialize and fit the logistic regression model.'''
        self.model_logistic = LogisticRegression()
        self.model_logistic.fit(self.X_train, self.y_train)
    
    def model_predict_linear(self):
        '''Predict using the linear model and return accuracy, precision, recall, f1, and support.'''
        self.model_fit_linear()
        assert self.model_linear is not None, "Model not initialized. Call model_fit_linear first."
        assert self.training_set is not None, "Training data not initialized."

        if self.perform_test and self.X_test is not None:
            predictions = np.round(self.model_linear.predict(self.X_test))
            accuracy = accuracy_score(self.y_test, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, predictions, average=None)
            assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
            return [accuracy, precision, recall, f1, support]

    def model_predict_logistic(self):
        '''Predict using the logistic model and return accuracy, precision, recall, f1, and support.'''
        self.model_fit_logistic()
        assert self.model_logistic is not None, "Model not initialized. Call model_fit_logistic first."
        assert self.training_set is not None, "Training data not initialized."

        if self.perform_test and self.X_test is not None:
            predictions = self.model_logistic.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, predictions, average=None)
            assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
            return [accuracy, precision, recall, f1, support]

    def matplot(self, model_type="logistic"):
        '''Visualize the decision boundary for logistic or linear model.'''
        if model_type == "logistic":
            self.model_fit_logistic()
            plot_decision_boundary(self.X_train, self.y_train, self.model_logistic)
        elif model_type == "linear":
            self.model_fit_linear()
            plot_decision_boundary(self.X_train, self.y_train, self.model_linear)

# Plot Decision Boundary Function
def plot_decision_boundary(X, y, model):
    '''Plot the decision boundary for any model.'''
    X = X.values if isinstance(X, pd.DataFrame) else X
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    if isinstance(model, LinearRegression):
        Z = np.round(Z)
    
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    plt.title('Decision Boundary')
    plt.xlabel('Exam Score 1')
    plt.ylabel('Exam Score 2')
    plt.show()

# Main function with argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear/Logistic Regression')
    parser.add_argument('-d', '--dataset_num', type=str, default="1", choices=["1", "2"], help='Dataset number: 1 or 2')
    parser.add_argument('-t', '--perform_test', action='store_true', help='Boolean flag for testing mode')
    args = parser.parse_args()
    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)

    linear_results = classifier.model_predict_linear()
    print("Linear Regression Results:", linear_results)
    logistic_results = classifier.model_predict_logistic()
    print("Logistic Regression Results:", logistic_results)

    # matplot
    print("\nVisualizing decision boundaries...\n")
    classifier.matplot(model_type="logistic")
    classifier.matplot(model_type="linear")