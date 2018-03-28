import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation

#### generate some x,y,z data ####
r = np.linspace(0,6,  num=100)
phi = np.linspace(0, 2*np.pi, num=200)
R, Phi = np.meshgrid(r,phi)
x = R*np.cos(Phi)
y = R*np.sin(Phi)
z = R
##################################

fig, ax=plt.subplots()
ax.set_aspect("equal")

def update(i):
    ax.clear()
    f = lambda x,y, offs, width, i: 1-i*np.exp(-(np.arctan2(x,y)-offs)**2/width)
    z_deformed = z*f(x,y, np.pi/4, 1./8., i=i)
    ax.contour(x,y,z_deformed, 10, linewidths=4)
    ax.contourf(x,y,z_deformed, 10, alpha=0.3)
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])

update(0) #plot the original data
anipath = 0.5*np.sin(np.linspace(0, np.pi, num=20))**2
ani = matplotlib.animation.FuncAnimation(fig, update, frames=anipath, interval = 100)

plt.show()