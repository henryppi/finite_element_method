import time

#disable sparse/obsolete warning
import warnings
warnings.filterwarnings("ignore")

from my_fem_functions import *

# PRE-PROCESSING

eSize = 0.002
xSize = 1*eSize
ySize = eSize

gauss_order = 2
L = 0.03#1
H = 0.02
F = -100
thickness = 0.005
nu = 0.288
E = 206.94e9
scaleDeformation = 0.1

bbox_support =   np.array([0, 0, 0, H],float)
bbox_load =   np.array([L, H, L, H],float)


D = (E/(1.0-nu**2))*np.array([[ 1.0, nu,  0.0         ],\
                    [ nu,  1.0, 0.0         ],\
                    [ 0.0, 0.0, 0.5*(1.0-nu)]])

print('meshing ...', end = '')
t_start = time.process_time() 
nodes,elements,nx,ny = makeMesh(L,H,xSize,ySize)
nElem = elements.shape[0]
nNode = nodes.shape[0]
nDof = 2*nNode

t_mesh = time.process_time() -t_start
print('\tdone in {:.3f} [s]'.format(t_mesh))

print('#Nodes \t\t{}'.format(nNode))
print('#Elements \t{}'.format(nElem))
print("#DoF \t\t{}".format(nDof))


t_start = time.process_time() 

gp,gw = gaussPointsQuad(gauss_order)
ngp =gp.shape[0]

indXY = np.zeros([8])
rows = []
cols = []
vals = []
for m in range(nElem):
    ind = elements[m,:]
    X = nodes[ind,:]

    Kloc = np.zeros([8,8],float)
    bloc = np.zeros([8,1],float)
    for ip in range(ngp):
        Nrs = shapeFunQ4(gp[ip,0],gp[ip,1])
        dNrs = shapeFunGradQ4(gp[ip,0],gp[ip,1])
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

print('assembly ...', end = '')
t_assem = time.process_time()-t_start
print('\tdone in {:.3f} [s]'.format(t_assem))

# constraints
ind_support = nodeSelectBBox2D(nodes,bbox_support,0.5*eSize)
dof_support = np.array([],int)
dof_support = np.append(dof_support,2*ind_support)
dof_support = np.append(dof_support,2*ind_support+1)

Kglob = apply_fixed_constraint(Kglob,dof_support)

# loads
ind_load = nodeSelectBBox2D(nodes,bbox_load,0.5*eSize)
dof_load = 2*ind_load+1

bglob = np.zeros([nDof,1],float)
bglob[dof_load,0] = F

plt.spy(Kglob.todense())
plt.show()

# SOLVE 
print('solving ... ', end = '')
t_start = time.process_time()
uSol = scipy.sparse.linalg.spsolve(Kglob, bglob)

displacement = uSol.reshape([nNode,2])
maxDisp = np.max(np.sqrt(np.sum(displacement**2,axis=1)))
t_sol = time.process_time()-t_start
print('\tdone in {:.3f} [s]'.format(t_sol))

# POST PROCESSING
print('post ... ', end = '')
t_start = time.process_time()
epsilon,sigma,vonMises = post_processing(nodes,elements,uSol,D)

max_epsilon = np.max(epsilon,axis=0)
min_epsilon = np.min(epsilon,axis=0)
max_sigma = np.max(sigma,axis=0)
min_sigma = np.min(sigma,axis=0)
min_vonMises = np.min(vonMises)
max_vonMises = np.max(vonMises)



# plotting
visu(L,
     H,
     nodes,
     elements,
     nElem,
     displacement,
     maxDisp,
     scaleDeformation,
     ind_load,
     dof_load,
     bglob,
     dof_support,
     vonMises,
     'run_06_plane_cantilever_basic.png')
     
plt.show()

t_post = time.process_time()-t_start
print('\tdone in {:.3f} [s]'.format(t_post))


print('max displacement {:.2e}[m]'.format(maxDisp))
print('min strain  xx: {:.2e},\tyy: {:.2e},\txy: {:.2e}'.format(min_epsilon[0],min_epsilon[1],min_epsilon[2]))
print('max strain  xx: {:.2e},\tyy: {:.2e},\txy: {:.2e}'.format(max_epsilon[0],max_epsilon[1],max_epsilon[2]))
print('min stress  xx: {:.2e},\tyy: {:.2e},\txy: {:.2e}'.format(min_sigma[0],min_sigma[1],min_sigma[2]))
print('max stress  xx: {:.2e},\tyy: {:.2e},\txy: {:.2e}'.format(max_sigma[0],max_sigma[1],max_sigma[2]))
print('von Mises stress min {:.2e}, max {:.2e}'.format(min_vonMises,max_vonMises))



