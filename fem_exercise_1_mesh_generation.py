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

elements = np.array([[0,1,5,4]],int)

line_index = [3,7,11]

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

# red line
ax.plot(vertices[line_index,0],vertices[line_index,1],'-r',lw=2)

# black vertex points
ax.plot(vertices[:,0],vertices[:,1],linestyle='none',color='k',marker='.',markersize=5)

# text at vertices 
for i in range(nVertices):
    ax.text(vertices[i,0],vertices[i,1], ' V'+str(i),fontsize=12,color='k')
plt.show()