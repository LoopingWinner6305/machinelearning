from pathlib import Path
import numpy as np
from ml_fundamentals.heart import load_data,make_preprocessor,evaluate_heart
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'Decision_trees'/'processed.cleveland.data'

def test_dataset_shape_and_missing_values():
    X,y=load_data(DATA)
    assert X.shape==(303,13) and int(X.isna().sum().sum())==6
    assert set(y.unique())=={0,1}

def test_unseen_category_is_handled_without_refitting():
    X,_=load_data(DATA);train=X.iloc[:20].copy();test=X.iloc[[20]].copy()
    transform=make_preprocessor();a=transform.fit_transform(train)
    test['cp']=999.;test['age']=np.nan
    b=transform.transform(test)
    assert b.shape==(1,a.shape[1]) and np.isfinite(b).all()

def test_holdout_does_not_enter_fitted_search():
    result,search,_,_=evaluate_heart(DATA)
    assert search.best_estimator_.named_steps['model'].tree_.n_node_samples[0]==result['train_rows']
    assert result['train_rows']+result['test_rows']==303
    for score in result['models'].values():
        assert sum(map(sum,score['confusion_matrix']))==result['test_rows']
        assert 0<=score['balanced_accuracy']<=1
