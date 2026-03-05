import numpy as np
import math

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    N=10000
    pos=np.ndarray((seq_length,d_model))
    for i in range (seq_length):
        for j in range(d_model):
            if j %2==0 :
                pos[i][j]=math.sin(i/N**((2*i)/d_model))
            else:
                pos[i][j]=math.cos(i/N**((2*i)/d_model)) 
    return pos