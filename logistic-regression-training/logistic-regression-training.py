import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n_rows=X.shape[0]
    features=X.shape[1]
    weights=np.zeros((features,))
    bias=np.array(0)
    for i in range(steps):
        z=np.dot(X,np.transpose(weights))+bias
        probs=_sigmoid(z)
        residuals=probs-y
        gradient=np.dot(X.T,residuals)/n_rows
        weights=weights-(lr*gradient)
        bias=bias-(lr*(residuals.mean()))
    return (weights,bias)
        
        
        
    
    
    
    