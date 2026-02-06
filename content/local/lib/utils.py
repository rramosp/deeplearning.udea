import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt

def sigmoide(u):
    g = np.exp(u)/(1 + np.exp(u))
    return g
#Aprendizaje
def Gradiente(X2,y2,MaxIter = 100000, eta = 0.01):
    w = np.array([30,-40,-120])#np.ones(3).reshape(3, 1)
    N = len(y2)
    Error =np.zeros(MaxIter)
    Xent = np.concatenate((X2,np.ones((100,1))),axis=1)

    for i in range(MaxIter):
        tem = np.dot(Xent,w)
        tem2 = sigmoide(tem.T)-np.array(y2)
        Error[i] = np.sum(abs(tem2))/N
        tem = np.dot(Xent.T,tem2.T)
        wsig = w - eta*tem/N
        w = wsig
    return w, Error

def plot_perceptron_frontier():
    fig, ax = plt.subplots(1,1)
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    text = ['Class 1','Class 2']
    colors = ['silver', 'limegreen', 'y', 'm', 'r']
    X2 = X[:100][:,:2]
    y2 = y[:100]
    for i in range(2):
        ax.scatter(X2[y2==i,0], X2[y2==i,1], color=colors[i],label=text[i])
    w,_ = Gradiente(X2,y2,MaxIter = 10000)
    x1 = np.linspace(4,7,20)
    x2 = -(w[0]/w[1])*x1 - (w[2]/w[1])
    line1, = ax.plot(x1,x2,'b',label='boundary')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()
    plt.show()