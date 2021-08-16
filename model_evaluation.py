from pandas.core.frame import DataFrame
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, plot_confusion_matrix, accuracy_score
import typing
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


train_path = 'assets/cleaned/data_train.csv'
test_path = 'assets/cleaned/data_test.csv'

# Read datasets from csv
def reading_csv_to_df(path_1: str, path_2: str):
    df_income_train = pd.read_csv(path_1)
    df_income_test = pd.read_csv(path_2)
    return df_income_train, df_income_test

df_train, df_test = reading_csv_to_df(train_path, test_path)

# Full dataset
df_income = df_train.append(df_test)

#Resample target (income = 1)
#df_resample = df_income[df_income['income'] == 1]
#df_final = df_income.append(df_resample)

# Split dataset
X = df_income.drop(columns=['income'])
y = df_income['income']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2, stratify=y)

#RandomForest Classifier
#Model Tuning & Evaluation

#GridSearchCV
def searching_best_features(X_train: pd.DataFrame, y_train: pd.DataFrame): 
    #GridSearchCV
    rfc = RandomForestClassifier()
    param_grid = { 
        'n_estimators': [5,10,50,100,250],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [2,4,8,16,32,None]}
    cv = GridSearchCV(rfc, param_grid=param_grid, scoring='accuracy',cv= 5, n_jobs=-1)
    return cv.fit(X_train, y_train.values.ravel())

#CV results
def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    #mean_score = results.cv_results_['mean_test_score']
    #std_score = results.cv_results_['std_test_score']
    #params = results.cv_results_['params']
    #for mean,std,params in zip(mean_score,std_score,params):
    #    print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')

cv = searching_best_features(X_train, y_train)
display(cv)


#Final model and results (metrics + roc plot)
def evaluating_classifier(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    cv = RandomForestClassifier(n_estimators=250, max_depth=16, max_features='auto', criterion='gini').fit(X_train, y_train)
    #CrossValidation on training set
    scores = cross_val_score(cv, X_train, y_train, cv=5) # cv is the number of folds (k)
    print(scores)
    print("CV Training Accuracy: {:.2f}% (+/- {:.2f})".format(scores.mean() * 100, scores.std() * 100))
    #model prediction
    y_pred = cv.predict(X_test)
    #Confusion matrix
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(cv, X_test, y_test)
    plt.show()
    #Classification Report
    print(classification_report(y_test, y_pred))

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve and ROC area and AUC Curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RandomForestClassifier(hypertuning) Income prediction')
    plt.legend(loc="lower right")
    plt.show()

    #Accuracy Scores
    print("ROC-AUC Score: ", roc_auc_score(y_test, y_pred))
    roc_evolution = roc_auc_score(y_test, y_pred)
    train_score_evolution = cv.score(X_train, y_train)
    print("Evaluating the model on the training set yields an accuracy of {:.2f}%".format(train_score_evolution*100))
    test_score_evolution = cv.score(X_test, y_test)
    print("Evaluating the model on the testing set yields an accuracy of {:.2f}%".format(test_score_evolution*100))
    return roc_evolution, train_score_evolution, test_score_evolution

# Follow-up on Scores
train_score_evolution = []
test_score_evolution = []
roc_evolution = []
# Run 5 times the model (CV and fit)
for i in range(5):
    #Split dataset again (different random_state)
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=i, test_size=0.2, stratify=y)
    roc_evo, train_score_evo, test_score_evo = evaluating_classifier(X_train, y_train, X_test, y_test)
    roc_evolution.append(roc_evo)
    train_score_evolution.append(train_score_evo)
    test_score_evolution.append(test_score_evo)

#Plot Scores evolution
n_experim = [1,2,3,4,5]
plt.plot(n_experim, train_score_evolution, c='blue')
plt.plot(n_experim, test_score_evolution, c='red')
plt.plot(n_experim, roc_evolution, c='green')
plt.xlabel('Runs')
plt.ylabel('Accuracy Score')
plt.title('Scores Evolution')
