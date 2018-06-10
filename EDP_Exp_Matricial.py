import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')

def tridiag(a,b,c,N):
    A = np.zeros((N,N))

    np.fill_diagonal(A[:-1,1:],a)
    np.fill_diagonal(A,b)
    np.fill_diagonal(A[1:,:-1],c)

    return A

def Exacta(x,t):
    n = len(x)
    sol = []
    for i in range(n):
        sol.append((np.exp((-1)*(np.pi**2)*(t)))*np.sin(np.pi*x[i]))
    return sol

def ErrorAb(a,b):
    n = len(a)
    c = []
    for i in range(n):
        c.append(np.absolute(a[i] - b[i]))
    return c

def EDP_Exp_Matricial(c, L, T, h, k, f, a, b):

    r = c*k/h**2
    m = round(L/h) + 1
    n = round(T/k) + 1

    x = np.linspace(0, L, m)

    w = np.zeros((n,m))
    w[0] = f(x)
    w[:,0] = a(0)
    w[:,-1] = b(L)

    W = tridiag(r, 1-2*r, r, m-2)

    for i in range(n-1):
        w[i+1,1:-1] = np.dot(W, w[i,1:-1])
        w[i+1,1] += r*w[i,0]
        w[i+1,-2] += r*w[i,-1]

    return w

Mat = EDP_Exp_Matricial(1, 1, 0.5, 0.1, 0.0005, lambda x: np.sin(np.pi * x), lambda x: 0, lambda x:0)

Exa1 = Exacta(Mat[-1],0.5)
Err1 = ErrorAb(Mat[-1],Exa1)
d = {'x': x , 'Aproximacion': Mat[-1], 'Exacta': Exa1, 'Error': Err1}
df = pd.DataFrame(data=d)
df = df[['x','Aproximacion','Exacta','Error']]
print(df)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

x = np.linspace(0, 1, 11)
t = np.linspace(0, 1, 1001)

def init():
  ax.plot([], [])
  ax.grid()
  ax.set_ylim(0, 1)
  ax.set_ylabel("Distribucion en el eje Y ")
  ax.set_xlabel("Posición en la cuerda")
  ax.set_title("Longitud de la barra ")

def animate(i):
  del ax.lines[:]
  ax.plot(x, Mat[i], color="blue", label="aprox. $t = {0:.2f}$ seg.".format(t[i]))
  ax.legend()

animation.FuncAnimation(fig, animate, init_func=init, frames=1001, interval=200, repeat=True)
plt.show()
