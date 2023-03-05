
# %%
import numpy as np
from scipy.stats import norm
# %%
def R_matrix (m,mu,sigma,delta,lamda):

    R = np.zeros((2*m+1,2*m+1)) # R is a square matrix of size 2m+1 x 2m+1 showing the transition probabilities from state j to state k where j and k are integers from -m to m, inclusive and represent the number of tranitional states of the process
    for row_counter in range(2*m+1):
        for col_counter in range(2*m+1):
            j=row_counter-m
            k=col_counter-m
            R[row_counter,col_counter] = norm.cdf((k+delta-(1-lamda)*j-mu*lamda)/(sigma*lamda))-norm.cdf((k-delta-(1-lamda)*j-mu*lamda)/(sigma*lamda))
    return R
def P_matrix (R):
    '''
    This function takes the R matrix and returns the transition matrix P
    Both R and P are square matrices
    R is matrix of size 2m+1 x 2m+1 
    P is matrix of size 2m+2 x 2m+2,  the extra +1 is coming from the absorbing state
    So we need to add a row and column to R to get P
    '''
    # Adding row of zeros to R
    row_of_zeros=np.zeros(R.shape[1]) #row of zeros with the same number of columns of R
    P=np.vstack([R,row_of_zeros]) #add a row of zeros to R toward getting P, the transition matrix
    
    # Adding column to R to get P
    row_sums = np.sum(P, axis=1)
    new_col = 1 - row_sums
    new_col = new_col.reshape((-1, 1))
    P = np.hstack((P, new_col)) 
    return P
# %%
UCL=
LCL=
mu=0
sigma=1
delta=0.01
lamda=0.95
m=np.ceil((UCL-LCL)/delta) #TODO: double check this
R=R_matrix(m,mu,sigma,delta,lamda)
print(R)
# %%
P=P_matrix(R)
print(P)
#
p_intial=np.zeros(2*m+1)
p_intial[m]=1
# %%
ARL=