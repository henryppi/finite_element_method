import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

print('my first fem mesh')

height = 2
width = 3
esize = 0.1
nx = 3
ny = 2

x = np.linspace(0,width,nx+1)
y = np.linspace(0,height,ny+1)

nVertices = x.shape[0]*y.shape[0]
xx,yy = np.meshgrid(x,y)

vertices = np.zeros([nVertices,2],float)
vertices[:,0] = xx.flatten()
vertices[:,1] = yy.flatten()

# elements = np.array([[0,1,5,4]],int)

elements = np.zeros([nx*ny,4],int)
for j in range(ny):
    ind = np.arange(nx)+nx*j
    elements[ind,0] = np.arange(0,nx)  +j*nx+j
    elements[ind,1] = np.arange(1,nx+1) +j*nx+j
    elements[ind,2] = np.arange(1,nx+1) +(j+1)*nx + j+1
    elements[ind,3] = np.arange(0,nx)   +(j+1)*nx + j+1

print(elements)
nElements = elements.shape[0]



fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

# filled  green polygons
pc = PolyCollection(vertices[elements],\
                    facecolor="#228B22",\
                    # facecolor=None,\
                    edgecolor="#16161D",\
                    alpha=0.5,\
                    linewidth=1)
ax.add_collection(pc)

# black vertex points
ax.plot(vertices[:,0],vertices[:,1],linestyle='none',color='k',marker='.',markersize=5)

# text at vertices 
for i in range(nVertices):
    ax.text(vertices[i,0],vertices[i,1], ' V'+str(i),fontsize=12,color='k')

# text at element center 
for i in range(nElements):
    cx = np.mean(vertices[elements[i,:],0])
    cy = np.mean(vertices[elements[i,:],1])
    ax.text(cx,cy, 'E'+str(i),fontsize=12,color='g')
plt.show()