from pandas.core.frame import DataFrame
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
import typing
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


train_path = 'assets/cleaned/data_train.csv'
test_path = 'assets/cleaned/data_test.csv'

# Read datasets from csv
def reading_csv_to_df(path_1: str, path_2: str) -> pd.DataFrame:
    df_income_train = pd.read_csv(path_1)
    df_income_test = pd.read_csv(path_2)
    return df_income_train, df_income_test

df_train, df_test = reading_csv_to_df(train_path, test_path)


#Merge datasets
df_income = df_train.merge(df_test, how='outer')

#Features and Target
X = df_income.drop(columns=['income'])
y = df_income['income']

#RandomForest Classifier
#Model Tuning & Evaluation

# Splitting Train-test RandomState
def spliting_data(X: pd.DataFrame, y:pd.Dataframe) -> pd.DataFrame:
    for random_state in range(4):
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=random_state, test_size=0.3)
        classifier = RandomForestClassifier(random_state=1)
        classifier.fit(X_train, y_train)
        score=classifier.score(X_test, y_test)
        print("Evaluating the model on the testing set yields an accuracy of {:.2f}% with random state {}".format(score*100, random_state))
    return X_train, X_test, y_train, y_test
    
X_train, X_test, y_train, y_test = spliting_data(X,y)


# CrossValidation
classifier = RandomForestClassifier(random_state=2)
for k in range(5,10):
    scores = cross_val_score(classifier, X, y, cv=k) # cv is the number of folds (k)
    print(scores)
    print("Accuracy: {:.2f}% (+/- {:.2f})".format(scores.mean() * 100, scores.std() * 100))


#GridSearchCV
def searching_best_features(X_train: pd.DataFrame, y_train: pd.DataFrame): 
    param_grid = { 
        'n_estimators': [100,800],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    classifier_cv = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy',cv= 7, verbose=0, n_jobs=-1)
    return classifier_cv.fit(X_train, y_train)

result = searching_best_features(X_train, y_train)
random_forest = result.best_estimator_
score = random_forest.score(X_test, y_test)
print("The best parameters are :", result.best_params_)
print("The best accuracy is {:.2f}%:".format(result.best_score_ * 100))
print("The generalization accuracy of the model is {:.2f}%".format(score * 100))
