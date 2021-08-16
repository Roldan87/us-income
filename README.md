# Model evaluation challenge - US Income

![Dollar Bill (Image)](assets/one_dollar_bill.jpg)


## The Data

The datasets `data_train.csv` and `data_test.csv` 


## The Mission

*Are you able to predict the income of every US citizen?*

### Constraints

- You must use `RandomForestClassifier()` from `sklearn`.
- Create **functions**, do **not** create a single huge script
- Each **function or class** has to be typed
- Each **function or class** has to contain a docstring in a [consistent format](https://stackoverflow.com/a/24385103).
- Your code should be **commented**.
- Your code should be **cleaned of any commented unused code**.

### The Deliverables

- Baseline accuracy
- Multiple evaluation metrics
- Hyper parameter tuning
- Some type of validation strategy

## US Income Prediction & Model Evaluation

### Installation
#### Python version
* Python 3.9
#### Packages
* Numpy
* Pandas
* Sklearn

### Usage

| File                | Description                             |
|---------------------|-----------------------------------------|
| model_evaluation.py | Python file containing *functions* for:<br>-Model Fitting<br>-Model Tuning<br>-Evaluation |
| assets              | Folder containing:<br>-datasets<br>-visuals |


### RandomForestClassifier()

#### Step 1. Baseline Accuracy

* Score<br/>
Evaluating the model on the training set yields an accuracy of 99.99%<br/>
Evaluating the model on the testing set yields an accuracy of 85.30%

**Note: The model shows overfitting**

* Confusion Matrix<br/>
[[11559   876]<br/>
 [ 1518  2328]]

* Classification Report<br/>
 ![classif report (Image)](assets/default_report.PNG)
 
 * ROC-AUC Curve<br/>
 ![roc curve(Image)](assets/default_roc_plot.png)
 
 * ROC-AUC Score: 0.7674
 
#### Step 2. Model Tuning

#### Random State (train-test split)

* random_state = [1,2,3,4]
* Results:<br/>
Evaluating the model on the testing set yields an accuracy of 85.76% with random state 0<br/>
Evaluating the model on the testing set yields an accuracy of 85.71% with random state 1<br/>
Evaluating the model on the testing set yields an accuracy of **85.99%** with **random state 2**<br/>
Evaluating the model on the testing set yields an accuracy of 85.73% with random state 3<br/>


##### GridSearchCV

* Param_grid ={<br/>'n_estimators': [5,10,50,100,250],<br/> 'criterion': ['gini', 'entropy'],<br/> 'max_features': ['auto', 'sqrt', 'log2'],<br/> 'max_depth': [2,4,8,16,32,None]<br/>}
* **Best Parameters: {criterion: 'gini','max_features': 'auto', 'n_estimators': 250, 'max_depth: 16}**

#### Step 3: Model Fit & Prediction
##### Model Run 5 times (Including CrossValidation)

**1st Run**
<p align='center'>
    <img src='assets/run_01.PNG' width='600'>
<p/>

<p align='center'>
    <img src='assets/run_01b.PNG' width='600'>
<p/>

**2nd Run**
<p align='center'>
    <img src='assets/run_02.PNG' width='600'>
<p/>

<p align='center'>
    <img src='assets/run_02b.PNG' width='600'>
<p/>

**3rd Run**
<p align='center'>
    <img src='assets/run_03.PNG' width='600'>
<p/>

<p align='center'>
    <img src='assets/run_03b.PNG' width='600'>
<p/>

**4th Run** 
<p align='center'>
    <img src='assets/run_04.PNG' width='600'>
<p/>

<p align='center'>
    <img src='assets/run_04b.PNG' width='600'>
<p/>

**5th Run** 

<p align='center'>
    <img src='assets/run_05.PNG' width='600'>
<p/>

<p align='center'>
    <img src='assets/run_05b.PNG' width='600'>
<p/>

### Evolution of Scores<br/>
**Legend:**<br/>
<span style="color: blue;">**Blue = Training Set**</span><br/>
<span style="color: red;">**Red = Test Set**</span><br/>
<span style="color: green;">**Green = ROC-AUC Score**</span><br/>
<p align="center">
    <img src="assets/score_evo.PNG" width='820' height='560'>
</p>

#### Step 4: Resampling
**Target samples unbalanced**<br/>
Oversampled ('income' = 1)<br/>
<p align='center'>
    <img src='assets/oversample_run.PNG' width='600'>
<p/>

### Final Conclusions & Results

* The Initial model showed:<br/>
**Training set score: 99.99% (overfitting)**<br/>
**Test set score: 85.30%**<br/>
* The Final model after tuning shows:<br/>
**Max Training set score: 90.19%**<br/>
**Max Test set score: 86.81%**<br/>
**Max ROC-AUC Score: 0.7763**<br/>
* The Resampled model shows:<br/>
**Max Training set score: 89.57%**<br/>
**Max Test set score: 86.39%**<br/>
**Max ROC-AUC Score: 0.86**<br/>

#### Evaluation Techniques used:<br/>
**train_test_split** - random_state=2, test_size=0.2, stratify=y<br/>
**GridSearchCV** - best_params_<br/>
![best parameters (Image)](assets/best_param.PNG)
**CrossValidation** - kfold = [4-8]<br/>
**Resampling** - target ('income'=1)<br/>
**Metrics** - Confusion Matrix / Classification Report / accuracy_score / ROC Curve / ROC-AUC score

#### Final Word<br/>
After re-splitting dataset and applying the evaluation methods (hyperparameter tuning, Cross Validation, GridSearchCV) the model showed: 
* No more **overfitting** in the training set. 
* The **accuracy score** in the *test set* only improved a bit, as well as the roc_auc_score.
* The model can accurately predict *True Negative* (target = 0) but has more difficulty to predict *True Positive* (target = 1).

After **oversample** the target, the results showed:
* Better Prediction Accuracy for both *True Negative* (0) and *True Positive* (1).
* The *training set* accuracy decreased.
* The *test set* accuracy improved compared to baseline but not compare to first model.
* The roc_auc_score improved quite a lot.

Excellent **BeCode** challenge to gain knowledge and sharpen skills regarding **model evaluation for machine learning systems**.






