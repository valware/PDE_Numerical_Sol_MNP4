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

def EDP_CN_Matricial(c, L, T, h, k, f, a, b):
    
    r = c*k/h**2
    m = round(L/h) + 1
    n = round(T/k) + 1
    
    x = np.linspace(0, L, m)
    
    w = np.zeros((n,m))
    w[0] = f(x)
    w[:,0] = a(0) 
    w[:,-1] = b(L)
    
    A = tridiag(-r/2, 1+r, -r/2, m-2)
    B = tridiag(r/2, 1-r, r/2, m-2)

         
    A_inv = np.linalg.inv(A)


    for i in range(n-1):
        
        C = np.dot(B, w[i,1:-1])
        C[0]  += r/2*(w[i+1,0]  + w[i,0])
        C[-1] += r/2*(w[i+1,-1] + w[i,-1])

        w[i+1,1:-1] = np.dot(A_inv, C)
    
    return w

Mat3 = EDP_CN_Matricial(1, 1, 0.5, 0.1, 0.0005, lambda x: np.sin(np.pi * x), lambda x: 0, lambda x:0)
Exa3 = Exacta(Mat3[-1],0.5)
Err3 = ErrorAb(Mat3[-1],Exa3)
d3 = {'x': x , 'Aproximacion': Mat3[-1], 'Exacta': Exa3, 'Error': Err3}
df3 = pd.DataFrame(data=d3)
df3 = df3[['x','Aproximacion','Exacta','Error']]
print(df3)

fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)

x3 = np.linspace(0, 1, 11)
t3 = np.linspace(0, 1, 1001)

def init3():
  ax3.plot([], [])
  ax3.grid()
  ax3.set_ylim(0, 1)
  ax3.set_ylabel("Distribucion en el eje Y ")
  ax3.set_xlabel("Posición en la cuerda")
  ax3.set_title("Longitud de la barra ")

def animate3(i):
  del ax3.lines[:]
  ax3.plot(x3, Mat3[i], color="blue", label="aprox. $t = {0:.2f}$ seg.".format(t[i]))
  ax3.legend()

animation.FuncAnimation(fig3, animate3, init_func=init3, frames=1001, interval=120, repeat=True)
plt.show()
