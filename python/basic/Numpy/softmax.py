"""Softmax."""
import numpy as np
import matplotlib.pyplot as plt

scores = [3.0, 1.0, 0.2]
#scores = np.array([[1, 2, 3, 6],[2,4,5,6],[3,8,7,6]])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    #if isinstance(scores,list) or x.ndim==1:
    emp_sum = np.exp(x).sum()
    return np.exp(x)/np.sum(np.exp(x),axis=0)
    #else:
    sums = []
    for col in range(x.shape[1]):
        sums.append(0)
    for col in range(x.shape[1]):
        for row in range(x.shape[0]):
            sums[col] +=  np.exp(x[row][col])

    #Calculate softmax for each value
    exp_sums = np.zeros(shape=(x.shape[0],x.shape[1]))
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            exp_sums[row][col] = float(np.exp(x[row][col]))/sums[col]
            #print np.exp(x[row][col]),"/",sums[col],"=", exp_sums[row][col]
    return exp_sums


#print(softmax(scores))

# Plot softmax curve
x = np.arange(-2.0, 6.0, 0.1)
print x.shape[0]," X ",x.ndim
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
print scores.shape[0]," X ",scores.shape[1]
print scores
transp = softmax(scores).T
print transp.shape[0]," X ",transp.shape[1]
print transp
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()