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

def EDP_Imp_Matricial(c, L, T, h, k, f, a, b):

    r = c*k/h**2
    m = round(L/h) + 1
    n = round(T/k) + 1

    x = np.linspace(0, L, m)

    w = np.zeros((n,m))
    w[0] = f(x)
    w[:,0] = a(0)
    w[:,-1] = b(L)

    W = tridiag(-r, 1+2*r, -r, m-2)

    W_inv = np.linalg.inv(W)

    for i in range(n-1):
        B = w[i,1:-1].copy()
        B[0] += r*w[i,0]
        B[-1] += r*w[i,-1]
        w[i+1,1:-1] = np.dot(W_inv, B)

    return w

Mat2 = EDP_Imp_Matricial(1, 1, 0.5, 0.1, 0.0005, lambda x: np.sin(np.pi * x), lambda x: 0, lambda x:0)
Exa2 = Exacta(Mat2[-1],0.5)
Err2 = ErrorAb(Mat2[-1],Exa2)
d2 = {'x': x , 'Aproximacion': Mat2[-1], 'Exacta': Exa2, 'Error': Err2}
df2 = pd.DataFrame(data=d2)
df2 = df2[['x','Aproximacion','Exacta','Error']]
print(df2)

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)

x2 = np.linspace(0, 1, 11)
t2 = np.linspace(0, 1, 1001)

def init2():
  ax2.plot([], [])
  ax2.grid()
  ax2.set_ylim(0, 1)
  ax2.set_ylabel("Distribucion en el eje Y ")
  ax2.set_xlabel("Posición en la cuerda")
  ax2.set_title("Longitud de la barra ")

def animate2(i):
  del ax2.lines[:]
  ax2.plot(x2, Mat2[i], color="blue", label="aprox. $t = {0:.2f}$ seg.".format(t[i]))
  ax2.legend()
  
animation.FuncAnimation(fig2, animate2, init_func=init2, frames=1001, interval=200, repeat=True)
plt.show()
