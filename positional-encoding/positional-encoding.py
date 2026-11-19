import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    pos=[[0]*d_model for _ in range(seq_len)]
    for i in range(seq_len):
        j=0
        while j<(d_model//2):
            pos[i][2*j]=np.sin(i/(base**(2*j/d_model)))
            pos[i][(2*j)+1]=np.cos(i/base**(2*j/d_model))
            j+=1
        if d_model%2!=0:
            pos[i][2*j]=np.sin(i/base**(2*j/d_model))
    return np.array(pos)