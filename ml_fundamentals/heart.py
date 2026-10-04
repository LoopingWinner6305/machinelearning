"""A reproducible educational comparison on the processed Cleveland dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

COLUMNS = ['age','sex','cp','restbp','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','hd']
CATEGORICAL = ['sex','cp','fbs','restecg','exang','slope','thal']
NUMERIC = [c for c in COLUMNS if c not in CATEGORICAL + ['hd']]

def load_data(path):
    frame = pd.read_csv(Path(path), header=None, names=COLUMNS, na_values='?')
    X = frame.drop(columns='hd')
    y = (frame.hd > 0).astype(int)
    return X, y

def make_preprocessor():
    numeric = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    categorical = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                            ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    return ColumnTransformer([('numeric',numeric,NUMERIC),('categorical',categorical,CATEGORICAL)])

def evaluate_heart(path):
    X, y = load_data(path)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    # Predeclare this grid. Do not use holdout scores to select candidates.
    pipeline = Pipeline([('preprocess',make_preprocessor()),('model',DecisionTreeClassifier(random_state=42))])
    folds = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    search = GridSearchCV(pipeline, {'model__ccp_alpha':[0.0,0.002,0.005,0.01,0.02,0.04,0.08]},
                          cv=folds,scoring='balanced_accuracy',n_jobs=1,refit=True)
    search.fit(X_train,y_train)
    models = {'majority_baseline':DummyClassifier(strategy='most_frequent'),
              'logistic_baseline':Pipeline([('preprocess',make_preprocessor()),
                                           ('model',LogisticRegression(max_iter=2000,random_state=42))]),
              'pruned_tree':search.best_estimator_}
    scores = {}
    for name,model in models.items():
        if name != 'pruned_tree':model.fit(X_train,y_train)
        pred = model.predict(X_test)
        scores[name] = {'accuracy':float(accuracy_score(y_test,pred)),
                       'balanced_accuracy':float(balanced_accuracy_score(y_test,pred)),
                       'f1':float(f1_score(y_test,pred,zero_division=0)),
                       'confusion_matrix':confusion_matrix(y_test,pred,labels=[0,1]).tolist()}
    result = {'rows':len(X),'train_rows':len(X_train),'test_rows':len(X_test),
              'missing_cells':int(X.isna().sum().sum()),'seed':42,
              'selected_ccp_alpha':float(search.best_params_['model__ccp_alpha']),
              'cv_balanced_accuracy':float(search.best_score_),
              'cv_fold_std':float(search.cv_results_['std_test_score'][search.best_index_]),
              'models':scores}
    return result,search,X_test,y_test
