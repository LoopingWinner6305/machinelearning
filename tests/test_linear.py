import numpy as np
import pytest
from ml_fundamentals.linear import sigmoid,binary_log_loss_from_logits,logistic_gradient,fit_logistic,predict_logistic,fit_perceptron

def test_sigmoid_extremes_are_finite():
    with np.errstate(over='raise',invalid='raise'):
        values = sigmoid(np.array([-1000.,0.,1000.]))
    np.testing.assert_allclose(values,[0.,0.5,1.])

def test_loss_handles_extreme_correct_and_wrong_predictions():
    assert binary_log_loss_from_logits(np.array([0,1]),np.array([-1000.,1000.])) == 0
    assert binary_log_loss_from_logits(np.array([1,0]),np.array([-1000.,1000.])) == 1000

def test_gradient_matches_independent_finite_difference():
    X=np.array([[1.,-1.,2.],[1.,2.,-0.5],[1.,0.3,1.]])
    y=np.array([0.,1.,1.]); w=np.array([0.1,-0.2,0.4]); eps=1e-6
    numerical=[]
    for i in range(len(w)):
        delta=np.zeros_like(w);delta[i]=eps
        numerical.append((binary_log_loss_from_logits(y,X@(w+delta))-binary_log_loss_from_logits(y,X@(w-delta)))/(2*eps))
    np.testing.assert_allclose(logistic_gradient(X,y,w),numerical,atol=1e-7)

def test_logistic_learns_simple_separable_data():
    X=np.array([[-2.],[-1.],[1.],[2.]]); y=np.array([0,0,1,1])
    w,h,_=fit_logistic(X,y,learning_rate=0.1,max_iterations=300)
    assert h[-1]['loss'] < h[0]['loss']
    np.testing.assert_array_equal(predict_logistic(X,w),y)

def test_iteration_budget_does_not_claim_convergence():
    _,h,converged=fit_logistic(np.array([[-1.],[1.]]),np.array([0,1]),max_iterations=1,tolerance=1e-12)
    assert len(h)==1 and not converged

def test_perceptron_stops_when_no_mistakes_remain():
    X=np.array([[-2.,3.],[0.,1.],[2.,-1.],[-2.,1.],[0.,-1.],[2.,-3.]])
    y=np.array([1,1,1,-1,-1,-1]);w,b,h=fit_perceptron(X,y)
    assert np.all(y*(X@w+b)>0) and h[-1]==0

def test_nonseparable_perceptron_respects_budget():
    _,_,h=fit_perceptron(np.array([[0.],[0.]]),np.array([-1,1]),max_epochs=5)
    assert len(h)==5 and h[-1]>0

def test_bad_labels_are_rejected():
    with pytest.raises(ValueError):binary_log_loss_from_logits(np.array([2]),np.array([0.]))
