import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import scipy.sparse.linalg
from my_modules import *

# case setup

width = 6
height = 3
force = -10.0

gauss_order = 2
thickness = 0.005
nu = 0.288
E = 206.94e9

scaleDeformation = 0.1
xdiv = 12
ydiv = 6
etype = 'quad4'
esize = 'ndiv'


rectangle = np.array([[0,0],[width,0],[width,height],[0,height]],float)

D = (E/(1.0-nu**2))*np.array([[ 1.0, nu,  0.0         ],\
                    [ nu,  1.0, 0.0         ],\
                    [ 0.0, 0.0, 0.5*(1.0-nu)]])

gp,gw = gauss_points_quad(gauss_order)
ngp =gp.shape[0]

# create mesh
vert,elem = create_meshgrid_q4(rectangle,esize,xdiv,ydiv)

nElem = elem.shape[0]
nVert = vert.shape[0]
nDof = 2*nVert

print('#Nodes \t\t{}'.format(nVert))
print('#Elements \t{}'.format(nElem))
print("#DoF \t\t{}".format(nDof))

# assemble stiffness matrix and load vector
indXY = np.zeros([8])
rows = []
cols = []
vals = []
for m in range(nElem):
    ind = elem[m,:]
    X = vert[ind,:]

    Kloc = np.zeros([8,8],float)
    bloc = np.zeros([8,1],float)
    for ip in range(ngp):
        Nrs = shape_fun_quad4(gp[ip,0],gp[ip,1])
        dNrs = shape_fun_quad4_grad(gp[ip,0],gp[ip,1])
        J = np.matrix(dNrs)*np.matrix(X)
        detJ = J[0,0]*J[1,1] - J[1,0]*J[0,1]
        invJ = (1.0/detJ)*np.matrix([[J[1,1],-J[0,1]],[-J[1,0],J[0,0]]])
        dNdX = invJ*dNrs
        B = np.matrix([[dNdX[0,0], 0.0, dNdX[0,1], 0.0, dNdX[0,2], 0.0, dNdX[0,3], 0.0],\
            [0.0, dNdX[1,0], 0.0, dNdX[1,1], 0.0, dNdX[1,2], 0.0, dNdX[1,3]],\
            [dNdX[1,0], dNdX[0,0], dNdX[1,1], dNdX[0,1], dNdX[1,2], dNdX[0,2], dNdX[1,3], dNdX[0,3]]])
        Kloc[:,:] += thickness*B.T*D*B*detJ*gw[ip];
    
    indXY[::2] = ind*2
    indXY[1::2] = ind*2+1
    r,c,v = mat2ijv(np.array(Kloc),indXY)
    
    rows += r
    cols += c
    vals += v

Kglob = coo_matrix((vals, (rows, cols)), shape=(nDof, nDof))
Kglob = Kglob.tolil()


# select fixed constraint vertices
neighbor = find_edge_neighbors(elem)
edge_internal,edge_external = get_internal_external_edges(elem,neighbor)

tol=0.01
boundary = []
edge_fix,edge_remain = select_boundary(vert,elem,edge_external,[0,0,0,height],tol)

# get index for constraint vertices
vert_ind_support = np.unique(edge_fix)
dof_support = np.array([],int)
dof_support = np.append(dof_support,2*vert_ind_support)
dof_support = np.append(dof_support,2*vert_ind_support+1)

# apply fixed constraints to DoFs
for i in dof_support:
    Kglob[:,i] = 0.
    Kglob[i,:] = 0.
    Kglob[i,i] = 1.

# load vector
vert_ind_load = node_select_bbox_2D(vert,np.array([width,height, width, height],float),tol)
vert_dof_load = 2*vert_ind_load+1

bglob = np.zeros([nDof,1],float)
bglob[vert_dof_load,0] = force

# plt.spy(Kglob.todense())
# plt.show()

uSol = scipy.sparse.linalg.spsolve(Kglob, bglob)
disp = uSol.reshape([nVert,2])
maxDisp = np.max(np.sqrt(np.sum(disp**2,axis=1)))

epsilon,sigma,vonMises = post_processing_global(vert,elem,uSol,D)

max_epsilon = np.max(epsilon,axis=0)
min_epsilon = np.min(epsilon,axis=0)
max_sigma = np.max(sigma,axis=0)
min_sigma = np.min(sigma,axis=0)
min_vonMises = np.min(vonMises)
max_vonMises = np.max(vonMises)

print('max displacement {:.2e}[m]'.format(maxDisp))
print('min strain  xx: {:.2e},\tyy: {:.2e},\txy: {:.2e}'.format(min_epsilon[0],min_epsilon[1],min_epsilon[2]))
print('max strain  xx: {:.2e},\tyy: {:.2e},\txy: {:.2e}'.format(max_epsilon[0],max_epsilon[1],max_epsilon[2]))
print('min stress  xx: {:.2e},\tyy: {:.2e},\txy: {:.2e}'.format(min_sigma[0],min_sigma[1],min_sigma[2]))
print('max stress  xx: {:.2e},\tyy: {:.2e},\txy: {:.2e}'.format(max_sigma[0],max_sigma[1],max_sigma[2]))
print('von Mises stress min {:.2e}, max {:.2e}'.format(min_vonMises,max_vonMises))

# plotting
visu(width,
     height,
     vert,
     elem,
     nElem,
     disp,
     maxDisp,
     scaleDeformation,
     vert_ind_load,
     vert_dof_load,
     bglob,
     dof_support,
     vonMises,
     'cantilever_quad4.png')
     
plt.show()