# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# ===== Ordinary Differential Equiations ===== #
def f1(y):
    dxdt = y
    return dxdt

def f2(w0,x,y,F,t,w,b):
    # Example function:  forced oscillator
    fun2 = -w0**2 * x - b*y + F*np.cos(w*t)
    return fun2

# ===== Methods ===== #
# Euler method
def euler(f1, f2, l, x, y, t, deltat, w0, w, F, b):
    """
    :param f1: function to solve 1
    :param f2: function to solve 2
    :param l: number of values
    :param x: initial value for x
    :param y: initial value for y
    :param t: initial value for t
    :param deltat: time step
    :param w0: param for funtion 2
    :param w: param for funtion 2
    :param F: param for funtion 2
    :param b: param for funtion 2
    :return: 3 lists -> estimated values for x, y for every value of t
    """
    xlist = np.zeros(l+1)
    ylist = np.zeros(l+1)
    tlist = np.zeros(l+1)
    xlist[0] = x
    ylist[0] = y
    tlist[0] = t
    for i in range(l):
        t = t + deltat
        xn = x + deltat * f1(y)
        yn = y + deltat * f2(w0, x, y, F, t, w, b)
        x = xn
        y = yn
        xlist[i+1] = x
        ylist[i+1] = y
        tlist[i+1] = t
    return xlist, ylist, tlist

# 2nd order Runge-Kutta method
def rungekuta2(f1, f2, l, x, y, t, deltat, w0, w, F, b):
    """
    :param f1: function to solve 1
    :param f2: function to solve 2
    :param l: number of values
    :param x: initial value for x
    :param y: initial value for y
    :param t: initial value for t
    :param deltat: time step
    :param w0: param for funtion 2
    :param w: param for funtion 2
    :param F: param for funtion 2
    :param b: param for funtion 2
    :return: 3 lists -> estimated values for x, y for every value of t
    """
    xlist = np.zeros(l+1)
    ylist = np.zeros(l+1)
    tlist = np.zeros(l+1)
    xlist[0] = x
    ylist[0] = y
    tlist[0] = t
    for i in range(l):
        t = t + deltat
        k1x = deltat * f1(y)
        k1y = deltat * f2(w0, x, y, F, t, w, b)
        k2x = deltat * f1(y + 0.5 * k1y)
        k2y = deltat * f2(w0, x + 0.5 * k1x, y + 0.5 * k1y, F, t + 0.5 * deltat, w, b)
        xn = x + k2x
        yn = y + k2y
        x = xn
        y = yn
        xlist[i+1] = x
        ylist[i+1] = y
        tlist[i+1] = t
    return xlist, ylist, tlist

# 4th order Runge-Kutta method
def rungekuta4(f1, f2, l, x, y, t, deltat, w0, w, F, b):
    """
    :param f1: function to solve 1
    :param f2: function to solve 2
    :param l: number of values
    :param x: initial value for x
    :param y: initial value for y
    :param t: initial value for t
    :param deltat: time step
    :param w0: param for funtion 2
    :param w: param for funtion 2
    :param F: param for funtion 2
    :param b: param for funtion 2
    :return: 3 lists -> estimated values for x, y for every value of t
    """
    xlist = np.zeros(l+1)
    ylist = np.zeros(l+1)
    tlist = np.zeros(l+1)
    xlist[0] = x
    ylist[0] = y
    tlist[0] = t
    for i in range(l):
        t = t + deltat
        k1x = deltat * f1(y)
        k1y = deltat * f2(w0, x, y, F, t, w, b)
        k2x = deltat * f1(y + 0.5 * k1y)
        k2y = deltat * f2(w0, x + 0.5 * k1x, y + 0.5 * k1y, F, t + 0.5 * deltat, w, b)
        k3x = deltat * f1(y + 0.5 * k2y)
        k3y = deltat * f2(w0, x + 0.5 * k2x, y + 0.5 * k2y, F, t + 0.5 * deltat, w, b)
        k4x = deltat * f1(y + k3y)
        k4y = deltat * f2(w0, x + k3x, y + k3y, F, t + deltat, w, b)
        xn = x + k1x/6. + k2x/3. + k3x/3. + k4x/6.
        yn = y + k1y/6. + k2y/3. + k3y/3. + k4y/6.
        x = xn
        y = yn
        xlist[i+1] = x
        ylist[i+1] = y
        tlist[i+1] = t
    return xlist, ylist, tlist


# ==== Examples ==== #

# Initial conditions
x=1.
y=1.
t=0.
deltat=0.01
F=1
b=1.
l=1000

# Different w0, w values

# Euler method
x1, y1, t1 = euler(f1,f2,l,x,y,t,deltat,1.,2.,F,b);x2,y2,t2=euler(f1,f2,l,x,y,t,deltat,1.,5.,F,b)
x3, y3, t3 = euler(f1,f2,l,x,y,t,deltat,5.,2.,F,b);x4,y4,t4=euler(f1,f2,l,x,y,t,deltat,5.,5.,F,b)

plt.figure(1)
plt.plot(t1,y1,'b-',linewidth=1,label='$\omega_0 = 1$ $\omega = 2$');plt.plot(t2,y2,'r-',linewidth=1,label='$\omega_0 = 1$ $\omega = 5$')
plt.plot(t3,y3,'g-',linewidth=1,label='$\omega_0 = 5$ $\omega = 2$');plt.plot(t4,y4,'y-',linewidth=1,label='$\omega_0 = 5$ $\omega = 5$')
plt.legend(loc='best')
plt.title('Oscilador armónico por el método de Euler')

# 2nd order Runge-Kutta method
x1,y1,t1=rungekuta2(f1,f2,l,x,y,t,deltat,1.,2.,F,b);x2,y2,t2=rungekuta2(f1,f2,l,x,y,t,deltat,1.,5.,F,b)
x3,y3,t3=rungekuta2(f1,f2,l,x,y,t,deltat,5.,2.,F,b);x4,y4,t4=rungekuta2(f1,f2,l,x,y,t,deltat,5.,5.,F,b)

plt.figure(2)
plt.plot(t1,y1,'b-',linewidth=1,label='$\omega_0 = 1$ $\omega = 2$');plt.plot(t2,y2,'r-',linewidth=1,label='$\omega_0 = 1$ $\omega = 5$')
plt.plot(t3,y3,'g-',linewidth=1,label='$\omega_0 = 5$ $\omega = 2$');plt.plot(t4,y4,'y-',linewidth=1,label='$\omega_0 = 5$ $\omega = 5$')
plt.legend(loc='best')
plt.title('Oscilador armónico por el método de Runge Kuta de 2° orden')

# 4th order Runge-Kutta method
x1,y1,t1=rungekuta4(f1,f2,l,x,y,t,deltat,1.,2.,F,b);x2,y2,t2=rungekuta4(f1,f2,l,x,y,t,deltat,1.,5.,F,b)
x3,y3,t3=rungekuta4(f1,f2,l,x,y,t,deltat,5.,2.,F,b);x4,y4,t4=rungekuta4(f1,f2,l,x,y,t,deltat,5.,5.,F,b)

plt.figure(3)
plt.plot(t1,y1,'b-',linewidth=1,label='$\omega_0 = 1$ $\omega = 2$');plt.plot(t2,y2,'r-',linewidth=1,label='$\omega_0 = 1$ $\omega = 5$')
plt.plot(t3,y3,'g-',linewidth=1,label='$\omega_0 = 5$ $\omega = 2$');plt.plot(t4,y4,'y-',linewidth=1,label='$\omega_0 = 5$ $\omega = 5$')
plt.legend(loc='best')
plt.title('Oscilador armónico por el método de Runge Kuta de 4° orden')


# Different delta t values

# Euler method
x1,y1,t1=euler(f1,f2,1000,x,y,t,0.01,5.,5.,F,b);x2,y2,t2=euler(f1,f2,200,x,y,t,0.05,5.,5.,F,b)

plt.figure(4)
plt.plot(t1,y1,'b-',linewidth=1,label='$\Delta t = 0.01 s$');plt.plot(t2,y2,'r-',linewidth=1,label='$\Delta t = 0.05 s$')
plt.legend(loc='best')
plt.title('Variamos $\Delta t$ - Oscilador armónico por el método de Euler')

# 2nd order Runge-Kutta method

x1,y1,t1=rungekuta2(f1,f2,1000,x,y,t,0.01,5.,5.,F,b);x2,y2,t2=rungekuta2(f1,f2,50,x,y,t,0.21,5.,5.,F,b)

plt.figure(5)
plt.plot(t1,y1,'b-',linewidth=1,label='$\Delta t = 0.01 s$');plt.plot(t2,y2,'r-',linewidth=1,label='$\Delta t = 0.2 s$')
plt.legend(loc='best')
plt.title('Variamos $\Delta t$ - Oscilador armónico por el método de Runge Kuta 2° orden')

# 4th order Runge-Kutta method
x1,y1,t1=rungekuta4(f1,f2,1000,x,y,t,0.01,5.,5.,F,b);x2,y2,t2=rungekuta4(f1,f2,40,x,y,t,0.3,5.,5.,F,b)

plt.figure(6)
plt.plot(t1,y1,'b-',linewidth=1,label='$\Delta t = 0.01 s$');plt.plot(t2,y2,'r-',linewidth=1,label='$\Delta t = 0.3 s$')
plt.legend(loc='best')
plt.title('Variamos $\Delta t$ - Oscilador armónico por el método de Runge Kuta 4° orden')
