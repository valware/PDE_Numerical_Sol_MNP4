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


def ErrorAb(a,b):
    n = len(a)
    c = []
    for i in range(n):
        c.append(np.absolute(a[i] - b[i]))
    return c


def EDP_Hiperbolica_Mat(c, L, T, h, k, f, g, a, b):
    #f = U(x,0) , g = U_t(x,0) (Condiciones iniciales)
    #a = U(0,t) , b = U(l,t) ( Condiciones frontera)
    
    r = (c*k**2)/h**2
    n = round(T/k) + 1
    m = round(L/h) + 1

    x = np.linspace(0,L,m)
    w = np.zeros((n,m))
    w[0] = f(x)
   
    for j in range(1,m-1):
        w[1,j] = (1-r)*f(x[j]) + (r/2)*f(x[j+1]) + (r/2)*f(x[j-1]) + k*g(x[j])

    w[:,0] = a(0)
    w[:,-1] = b(L)

    W = tridiag(r, 2-2*r, r, m-2)

    for i in range(1,n-1):
        w[i+1,1:-1] = np.dot(W, w[i,1:-1]) - w[i-1,1:-1]
        w[i+1,1] += r*w[i,0]
        w[i+1,-2] += r*w[i,-1]
        
    return w

def Exacta2(x,t):
    n = len(x)
    sol = []
    for i in range(n):
        sol.append((np.sin(np.pi * x[i])) * (np.cos(2*np.pi * t)))
    return sol

HipH = EDP_Hiperbolica_Mat(4, 1, 1, 0.1, 0.05, lambda x: np.sin(np.pi * x), lambda x: 0, lambda x: 0, lambda x:0)
ExaH = Exacta2(HipH[-1],0.5)
ErrH = ErrorAb(HipH[-1],ExaH)
dH = {'x': x , 'Aproximacion': HipH[-1], 'Exacta': ExaH, 'Error': ErrH}
dH = pd.DataFrame(data=dH)
dH = dH[['x','Aproximacion','Exacta','Error']]
print(dH)

figH = plt.figure()
axH = figH.add_subplot(1,1,1)

xH = np.linspace(0, 1, 11)
tH = np.linspace(0, 1, 21)

def initH():
  axH.plot([], [])
  axH.grid()
  axH.set_ylim(-1, 1)
  axH.set_ylabel("ALTURA")
  axH.set_xlabel("Posición en la cuerda")
  axH.set_title("Movimiento de la cuerda")

def animateH(i):
  del axH.lines[:]
  axH.plot(xH, HipH[i], color="blue", label="aprox. $t = {0:.2f}$ seg.".format(tH[i]))
  axH.legend()

animation.FuncAnimation(figH, animateH, init_func=initH, frames=21, interval=120, repeat=True)
plt.show()
