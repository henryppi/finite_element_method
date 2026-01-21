import numpy as np
import matplotlib.pyplot as plt

from my_modules import *

width = 6
height = 3

rectangle = np.array([[0,0],[width,0],[width,height],[0,height]],float)

xdiv = 6
ydiv = 3
etype = 'quad4'
esize = 'ndiv'

vert,elem = create_meshgrid_q4(rectangle,esize,xdiv,ydiv)
# vert,elem = merge_nodes(vert,elem)
neighbor = find_edge_neighbors(elem)

edge_internal,edge_external = get_internal_external_edges(elem,neighbor)

tol=0.01
boundary = []
edge_fix,edge_remain = select_boundary(vert,elem,edge_external,[0,0,0,height],tol)
boundary.append(['fixed',edge_fix])
edge_force,edge_remain = select_boundary(vert,elem,edge_remain,[width,width,0,height],tol)
boundary.append(['force',edge_force])
boundary.append(['free',edge_remain])

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
ax = plot_mesh(ax,vert,elem,boundary)
ax = plot_mesh_numbers(ax,vert,elem,boundary)
ax.axis("equal")
ax.axis("off")
plt.savefig('./images/ass1_quad_mesh.png',bbox_inches='tight',dpi=200)
plt.show()