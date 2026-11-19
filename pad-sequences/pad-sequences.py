import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.array([],dtype=int).reshape(0,0)
    if max_len==None:
        l=float("-inf")
        for vec in seqs:
            l=max(l,len(vec))
        max_len=l
    ans=[]
    for s in seqs:
        if len(s)>=max_len:
            ans.append(s[:max_len])
        else:
            ans.append(s+[pad_value]*(max_len-len(s)))

    return np.array(ans,dtype=int)
            