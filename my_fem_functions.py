import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
import scipy.sparse.linalg

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def magnitude(v):
    return np.sqrt(np.sum(v**2))

def makeMesh(L,H,xSize,ySize):
    
    x = np.linspace(0,L,int(np.round(L/xSize))+1)
    y = np.linspace(0,H,int(np.round(H/ySize))+1)
    X,Y = np.meshgrid(x,y)
    
    nx = x.shape[0]
    ny = y.shape[0]
    
    nodes = np.zeros([nx*ny,2],float)
    nodes[:,0] = X.flatten()
    nodes[:,1] = Y.flatten()
    
    elements = np.zeros([(nx-1)*(ny-1),4])
    
    k = 0
    for j in range(ny-1):
        for i in range(nx-1):
            elements[k,:] = [i+j*nx, 1+i+j*nx, 1+i+(j+1)*nx,i+(j+1)*nx]
            k+=1;

    return nodes,elements.astype(int),nx,ny

def save_mesh(vert,elem,fn):
    fn_vert = fn+'.nod'
    fn_elem = fn+'.elm'
    np.savetxt(fn_vert, vert, delimiter=',')
    np.savetxt(fn_elem, elem, fmt='%i',delimiter=',')

def load_mesh(fn):
    fn_vert = fn+'.nod'
    fn_elem = fn+'.elm'
    vert = np.loadtxt(fn_vert,delimiter=',')
    elem = np.loadtxt(fn_elem,delimiter=',')
    return vert,elem.astype(int)
    
def convert3NodeTo6Node(elements,vertices):
    nElem = elements.shape[0]
    nVert = vertices.shape[0]
    i1 = elements[:,0]
    i2 = elements[:,1]
    i3 = elements[:,2]
    e1x = 0.5*(vertices[i2,0]+vertices[i3,0])
    e2x = 0.5*(vertices[i3,0]+vertices[i1,0])
    e3x = 0.5*(vertices[i1,0]+vertices[i2,0])
    e1y = 0.5*(vertices[i2,1]+vertices[i3,1])
    e2y = 0.5*(vertices[i3,1]+vertices[i1,1])
    e3y = 0.5*(vertices[i1,1]+vertices[i2,1])
    ex = np.concatenate([e1x,e2x,e3x])
    ey = np.concatenate([e1y,e2y,e3y])
    ex = ex.reshape([3,nElem]).T.flatten()
    ey = ey.reshape([3,nElem]).T.flatten()
    nEdge = ex.shape[0]
    elemInd = np.concatenate([range(nElem),range(nElem),range(nElem)])
    edgeInd = np.concatenate([0*np.ones(nElem),1*np.ones(nElem),2*np.ones(nElem)])

    XX = np.repeat(np.array([ex]).T,nEdge,axis=1)
    YY = np.repeat(np.array([ey]),nEdge,axis=0)
    DX = XX.T-XX
    DY = YY-YY.T
    RR = np.sqrt((DX)**2+(DY)**2)
    RR += np.eye(RR.shape[0],RR.shape[1])
    I,J = np.where(RR==0)
        
    nInd = I.shape[0]
    ind = np.zeros([nInd,2],int)
    ind[:,0] = I
    ind[:,1] = J
    ind.sort(axis=1)
    negI = np.delete(np.arange(nEdge),I)
    ind2 = np.unique(ind.view(np.dtype((np.void, ind.dtype.itemsize*ind.shape[1])))).view(ind.dtype).reshape(-1, ind.shape[1])
    newInd = ind2[:,0]
    oldInd = ind2[:,1]
    nInd2 = ind2.shape[0]
  
    edgeVert = np.zeros([3*nElem,2],float)
    edgeVert[:,0] = ex
    edgeVert[:,1] = ey
    nEdgeVert = edgeVert.shape[0]
    edgeInd = np.arange(nEdgeVert)
    indReplace = np.searchsorted(edgeInd,oldInd)
    edgeInd[indReplace] = newInd
    edgeElem = edgeInd.reshape(int(nEdgeVert/3),3)
    edgeElem += nVert
    
    newElem = np.concatenate([elements,edgeElem],axis=1)
    newVert = np.concatenate([vertices,edgeVert],axis=0)
    
    return newElem,newVert

def removeUnusedVertices(elements,vertices):
    nElem,nNode = elements.shape
    nVert,nDof = vertices.shape
    vertInd = elements.flatten() # row mayor
    sortInd = np.argsort(vertInd)
    sortSortInd = np.argsort(sortInd)
    vertIndSorted = vertInd[sortInd]
    shiftInd = np.diff(vertIndSorted)
    shiftInd = np.insert(shiftInd,0,0)-1
    shiftInd[np.where(shiftInd<1)] = 0
    shiftInd = np.cumsum(shiftInd)
    vertIndShifted = vertIndSorted-shiftInd
    newVertInd = vertIndShifted[sortSortInd]
    reducedInd = np.unique(vertIndSorted)
    newVert = vertices[reducedInd,:]
    newElem = np.reshape(newVertInd,[nElem,nNode])

    return newElem,newVert

def gaussPointsTri(order):
    if order==1:
        gw = np.array([1.0])
        gp = np.array([[1./3., 1./3.]])
    elif order==2:
        gw = np.array([1./3., 1./3., 1./3.])
        gp = np.array([[1./6., 1./6.],\
                       [2./3., 1./6.],\
                       [1./6., 2./3.]])
    elif order==3:
        gw = np.array([-27./48., 25./48., 25./48., 25./48.])
        gp = np.array([[1./3., 1./3.],\
                       [0.2, 0.2],\
                       [0.6, 0.2],\
                       [0.2, 0.6]])
    elif order==4: 
        gw = np.array([0.223381589678011,\
                       0.223381589678011,\
                       0.223381589678011,\
                       0.109951743655322,\
                       0.109951743655322,\
                       0.109951743655322])
        gp = np.array([[0.445948490915965, 0.445948490915965],\
                       [0.445948490915965, 0.108103018168070],\
                       [0.108103018168070, 0.445948490915965],\
                       [0.091576213509771, 0.091576213509771],\
                       [0.091576213509771, 0.816847572980459],\
                       [0.816847572980459, 0.091576213509771]])
    else:
        print("error triangle order to high")

    return gp,gw

def quadTriShapeFun(r,s):
    return np.array([1.0-3.0*r-3.0*s+2.0*r**2+4.0*r*s+2.0*s**2,\
                     2.0*r**2-r,\
                     2.0*s**2-s,\
                     4.0*r*s,\
                     4.0*s-4.0*r*s-4.0*s**2,\
                     4.0*r-4.0*r**2-4.0*r*s])

def quadTriShapeFunDeriv(r,s):
#     dNdrs = np.zeros([2,6])
#     dNdrs[0,:] = [-3.0+4.0*r+4.0*s, 4.0*r-1.0, 0.0, 4.0*s, -4.0*s, 4.0-8.0*r-4.0*s]
#     dNdrs[1,:] = [-3.0+4.0*r+4.0*s, 0.0, 4.0*s-1.0, 4.0*r, 4.0-4.0*r-8.0*s, -4.0*r]\
    return np.array([[-3.0+4.0*r+4.0*s,\
                      4.0*r-1.0,\
                      0.0,\
                      4.0*s,\
                      -4.0*s,\
                      4.0-8.0*r-4.0*s],\
                     [-3.0+4.0*r+4.0*s,\
                      0.0,\
                      4.0*s-1.0,\
                      4.0*r,\
                      4.0-4.0*r-8.0*s,
                      -4.0*r]])

def isoQuadTriStiff(X,Dmat,thick,order=3):
    Kmat = np.zeros([12,12],float)
    Fvec = np.zeros([12,1],float)
    gp,gw = gaussPointsTri(order)
    ngp = gp.shape[0]
    for ip in range(ngp):
        Nrs = quadTriShapeFun(gp[ip,0],gp[ip,1])
        dNrs = quadTriShapeFunDeriv(gp[ip,0],gp[ip,1])
        jac = np.matrix(dNrs)*X
        detJ = jac[0,0]*jac[1,1]- jac[0,1]*jac[1,0]
        invJ = (1.0/detJ)*np.matrix([[jac[1,1],-jac[0,1]],[-jac[1,0],jac[0,0]]])
        dNdX = invJ*dNrs
        B = np.matrix([[dNdX[0,0],0.0,dNdX[0,1],0.0,dNdX[0,2],0.0,dNdX[0,3],0.0,dNdX[0,4],0.0,dNdX[0,5],0.0],\
                       [0.0,dNdX[1,0],0.0,dNdX[1,1],0.0,dNdX[1,2],0.0,dNdX[1,3],0.0,dNdX[1,4],0.0,dNdX[1,5]],\
                       [dNdX[1,0],dNdX[0,0],dNdX[1,1],dNdX[0,1],dNdX[1,2],dNdX[0,2],dNdX[1,3],dNdX[0,3],dNdX[1,4],dNdX[0,4],dNdX[1,5],dNdX[0,5]]])
        Kmat += B.T*Dmat*B*detJ*gw[ip]*thick
    return Kmat,Fvec
    
def assembleGlobalStiffnessMatrix(vertices,elements,Dmat,thickness,order):
    nElem = elements.shape[0]
    nVert = vertices.shape[0]
    
    indXY = np.zeros([12])
    rows = []
    cols = []
    vals = []
        
    for m in range(nElem):
        ind = elements[m,:]
        X = vertices[ind,:]

        mat,Fvec = isoQuadTriStiff(X,Dmat,thickness,order)
        
        indXY[::2] = ind*2
        indXY[1::2] = ind*2+1
        r,c,v = mat2ijv(np.array(mat),indXY)
        
        rows += r
        cols += c
        vals += v


    stiffMat = coo_matrix((vals, (rows, cols)), shape=(2*nVert, 2*nVert))
    stiffMat = stiffMat.tolil()
    return stiffMat
    
def get_bbox(nodes):
    return [np.min(nodes[:,0]),np.min(nodes[:,1]),np.max(nodes[:,0]),np.max(nodes[:,1])]

def analyze_tri3_mesh(nodes,elements):
    nElem = elements.shape[0]
    
    eSizes = np.zeros([nElem,5])
    for i in range(nElem):
        v0 = nodes[elements[i,0],:]
        v1 = nodes[elements[i,1],:]
        v2 = nodes[elements[i,2],:]
        a = magnitude(v1-v0)
        b = magnitude(v2-v1)
        c = magnitude(v0-v2)
        s = 0.5*(a+b+c)
        abc = (s-a)*(s-b)*(s-c)
        ri = np.sqrt(abc/s)
        A = np.sqrt(s*abc)
        ro = a*b*c/A/4
        eSizes[i,:] = [a,b,c,2*ri,2*ro]
    eSize_min = np.min(eSizes[:,3])
    eSize_max = np.max(eSizes[:,4])
    eSize_mean = np.mean(0.5*(eSizes[:,4]+eSizes[:,3]))
    return eSize_min,eSize_max,eSize_mean

def nodeSelectBBox2D(vertices,bbox,border=0):
    bbox += [-border,-border,border,border]
    nVert,nDim = vertices.shape
    
    xminBool = np.zeros([nVert,1],bool)
    xmaxBool = np.zeros([nVert,1],bool)
    yminBool = np.zeros([nVert,1],bool)
    ymaxBool = np.zeros([nVert,1],bool)
    xmin = np.where(vertices[:,0]>=bbox[0])[0] 
    xmax = np.where(vertices[:,0]<=bbox[2])[0]        
    ymin = np.where(vertices[:,1]>=bbox[1])[0]
    ymax = np.where(vertices[:,1]<=bbox[3])[0]
    xminBool[xmin,0] = True
    xmaxBool[xmax,0] = True
    yminBool[ymin,0] = True
    ymaxBool[ymax,0] = True

    mask = xminBool & xmaxBool & yminBool & ymaxBool
    ind = np.where(mask==True)[0]
    return np.array([ind]).T.astype(int)

def apply_fixed_constraint(K,dof):
    for i in dof:
        K[:,i] = 0.
        K[i,:] = 0.
        K[i,i] = 1.
    return K
    
def gaussPointsQuad(order):
    if order==1:
        gw = np.array([4.0])
        gp = np.array([[0., 0.]])
    elif order==2:
        gw = np.array([1., 1., 1., 1.])
        gp = np.array([[-0.577350269189626, -0.577350269189626],\
              [ 0.577350269189626, -0.577350269189626],\
              [ 0.577350269189626,  0.577350269189626],\
              [-0.577350269189626,  0.577350269189626]])
    elif order==3:
        gw = np.array([0.555555555555556*0.555555555555556,\
              0.555555555555556*0.888888888888889, \
              0.555555555555556*0.555555555555556, \
              0.888888888888889*0.555555555555556, \
              0.888888888888889*0.888888888888889, \
              0.888888888888889*0.555555555555556, \
              0.555555555555556*0.555555555555556, \
              0.555555555555556*0.888888888888889, \
              0.555555555555556*0.555555555555556])
        gp = np.array([[-0.774596669241483, -0.774596669241483  ],\
              [-0.774596669241483,  0.0                ],\
              [-0.774596669241483,  0.774596669241483  ],\
              [ 0.0,               -0.774596669241483  ],\
              [ 0.0,                0.0                ],\
              [ 0.0,                0.774596669241483  ],\
              [ 0.774596669241483, -0.774596669241483  ],\
              [ 0.774596669241483,  0.0                ],\
              [ 0.774596669241483,  0.774596669241483 ]])
    else:
        print('gauss order {} not implemented'.format(order))

    return gp,gw

def mat2ijv(mat,ind):
    nRow,nCol = mat.shape 
    rows = np.tile(ind,nRow)
    cols = np.repeat(ind,nCol)
    vals = mat.ravel()
    return rows.tolist(),cols.tolist(),vals.tolist()


def shapeFunQ4(r,s):
    Nrs =  0.25*np.array([ (1 - r) * (1 - s),\
                  (1 + r) * (1 - s),\
                  (1 + r) * (1 + s), \
                  (1 - r) * (1 + s)])
    return Nrs

def shapeFunGradQ4(r,s):
    dNdX =  0.25*np.array([[-1 + s , 1 - s , 1 + s , -1 - s ],\
                  [-1 + r ,-1 - r , 1 + r ,  1 - r ]])
    return dNdX
    
def post_processing(nodes,elements,u_glob,D):
    gauss_order = 2
    nElem = elements.shape[0]
    epsilon = np.zeros([nElem,3],float)
    sigma = np.zeros([nElem,3],float)
    vonMises = np.zeros([nElem,1],float)

    gp,gw = gaussPointsQuad(gauss_order)
    ngp = gp.shape[0]

    ix = np.arange(0,8,2)
    iy = np.arange(0,8,2)+1
    dof_glob = np.zeros([8],int)
    u_loc = np.zeros([8,1],float)
        
    for m in range(nElem):
        ind = elements[m,:]
        X = nodes[ind,:]

        
        dof_glob[ix] = ind*2
        dof_glob[iy] = ind*2+1
        u_loc[:,0] = u_glob[dof_glob]
        
        epsilon_elem = np.zeros([3,1])
        sigma_elem = np.zeros([3,1])
        
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
            epsilon_elem += B*u_loc*detJ*gw[ip]
            sigma_elem += np.matrix(D)*epsilon_elem
        epsilon[m,:] = epsilon_elem.T
        sigma[m,:] = sigma_elem.T
        vonMises[m,0] = np.sqrt( sigma_elem[0,0]**2 + sigma_elem[1,0]**2 - sigma_elem[0,0]*sigma_elem[1,0] + 3*sigma_elem[2,0]**2 )
        
    return epsilon,sigma,vonMises
    
def post_processing_tri6(nodes,elements,u_glob,D):
    gauss_order = 2
    
    nElem = elements.shape[0]
    nNodes = nodes.shape[0] 
    nDof = 2*elements.shape[1]   

    ix = np.arange(0,nDof,2)
    iy = np.arange(0,nDof,2)+1
    dof_glob = np.zeros([nDof],int)
    u_loc = np.zeros([nDof,1],float)
    
    epsilon = np.zeros([nElem,3],float)
    sigma = np.zeros([nElem,3],float)
    vonMises = np.zeros([nElem,1],float)

        
    for m in range(nElem):
        ind = elements[m,:]
        X = nodes[ind,:]

        dof_glob[ix] = ind*2
        dof_glob[iy] = ind*2+1
        u_loc[:,0] = u_glob[dof_glob]
        
        epsilon_elem = np.zeros([3,1])
        sigma_elem = np.zeros([3,1])

        gp,gw = gaussPointsTri(gauss_order)
        ngp = gp.shape[0]
        for ip in range(ngp):
            Nrs = quadTriShapeFun(gp[ip,0],gp[ip,1])
            dNrs = quadTriShapeFunDeriv(gp[ip,0],gp[ip,1])
            jac = np.matrix(dNrs)*X
            detJ = jac[0,0]*jac[1,1]- jac[0,1]*jac[1,0]
            invJ = (1.0/detJ)*np.matrix([[jac[1,1],-jac[0,1]],[-jac[1,0],jac[0,0]]])
            dNdX = invJ*dNrs
            B = np.matrix([[dNdX[0,0],0.0,dNdX[0,1],0.0,dNdX[0,2],0.0,dNdX[0,3],0.0,dNdX[0,4],0.0,dNdX[0,5],0.0],\
                           [0.0,dNdX[1,0],0.0,dNdX[1,1],0.0,dNdX[1,2],0.0,dNdX[1,3],0.0,dNdX[1,4],0.0,dNdX[1,5]],\
                           [dNdX[1,0],dNdX[0,0],dNdX[1,1],dNdX[0,1],dNdX[1,2],dNdX[0,2],dNdX[1,3],dNdX[0,3],dNdX[1,4],dNdX[0,4],dNdX[1,5],dNdX[0,5]]])

    
            epsilon_elem += B*u_loc*detJ*gw[ip]
            sigma_elem += np.matrix(D)*epsilon_elem
        epsilon[m,:] = epsilon_elem.T
        sigma[m,:] = sigma_elem.T
        vonMises[m,0] = np.sqrt( sigma_elem[0,0]**2 + sigma_elem[1,0]**2 - sigma_elem[0,0]*sigma_elem[1,0] + 3*sigma_elem[2,0]**2 )
        
    return epsilon,sigma,vonMises
    
    
    nElem = elements.shape[0]
    nElemNodes = elements.shape[1]
    nDof = 2*nElemNodes
    
    epsilon = np.zeros([nElem,3],float)
    sigma = np.zeros([nElem,3],float)
    vonMises = np.zeros([nElem,1],float)

    gp,gw = gaussPointsQuad(gauss_order)
    ngp = gp.shape[0]

    ix = np.arange(0,8,2)
    iy = np.arange(0,8,2)+1
    dof_glob = np.zeros([8],int)
    u_loc = np.zeros([8,1],float)
        
    for m in range(nElem):
        ind = elements[m,:]
        X = nodes[ind,:]

        
        dof_glob[ix] = ind*2
        dof_glob[iy] = ind*2+1
        u_loc[:,0] = u_glob[dof_glob]
        
        epsilon_elem = np.zeros([3,1])
        sigma_elem = np.zeros([3,1])
        
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
            epsilon_elem += B*u_loc*detJ*gw[ip]
            sigma_elem += np.matrix(D)*epsilon_elem
        epsilon[m,:] = epsilon_elem.T
        sigma[m,:] = sigma_elem.T
        vonMises[m,0] = np.sqrt( sigma_elem[0,0]**2 + sigma_elem[1,0]**2 - sigma_elem[0,0]*sigma_elem[1,0] + 3*sigma_elem[2,0]**2 )
        
    return epsilon,sigma,vonMises
    

def visu_alt(L,
         H,
         nodes,
         elements,
         nElem,
         displacement,
         maxDisp,
         scaleDeformation,
         dof_load,
         bglob,
         dof_support,
         vonMises,fn):
    fig, (ax1,ax2) = plt.subplots(2, 1)
    fig.patch.set_visible(False)

    diag = np.sqrt((np.max(nodes[:,0])-np.min(nodes[:,0]))**2+(np.max(nodes[:,1])-np.min(nodes[:,1]))**2)
    nodes_deformed = nodes + (scaleDeformation*diag/maxDisp)*displacement

    # plot Mesh & BC
    patches = []
    for i in range(nElem):
        elem =elements[i,:]
        x = nodes[elem,0]
        y = nodes[elem,1]
        X = np.concatenate((np.array([x]).T,np.array([y]).T),axis=1)
        # polygon = Polygon(X, True)
        polygon = Polygon(X)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1.0,edgecolor='black',facecolor='lightgreen',linewidth=1)
    ax1.add_collection(p) 

    for i in range(len(dof_load)):
        if (dof_load[i] % 2) == 0:
            ax1.quiver(nodes[int((dof_load[i])/2),0], nodes[int((dof_load[i])/2),1],
                       [np.sign(bglob[dof_load[i],0])],[0],
                       color='red')
        else:
            ax1.quiver(nodes[int((dof_load[i]-1)/2),0], nodes[int((dof_load[i]-1)/2),1],
                       [0],[np.sign( bglob[dof_load[i],0] )], 
                       color='red')

    for i in range(len(dof_support)):
        if (dof_support[i] % 2) == 0:
            ax1.plot(nodes[int((dof_support[i])/2),0], nodes[int((dof_support[i])/2),1], 
            marker=5, 
            markersize=1/scaleDeformation, 
            fillstyle='full',
            markerfacecolor='blue',
            markeredgecolor='none')

        else:
            ax1.plot(nodes[int((dof_support[i]-1)/2),0], nodes[int((dof_support[i]-1)/2),1], 
            marker=6, 
            markersize=1/scaleDeformation, 
            fillstyle='full',
            markerfacecolor='blue',
            markeredgecolor='none')

    ax1.set_title('Mesh & BC')
    ax1.axis('off')
    ax1.set_xlim([0,L])
    ax1.set_ylim([0,H])
    ax1.axis('equal')

    # plot Displacement & von Mises stress
    patches = []
    for i in range(nElem):
        elem =elements[i,:]
        x = nodes_deformed[elem,0]
        y = nodes_deformed[elem,1]
        X = np.concatenate((np.array([x]).T,np.array([y]).T),axis=1)
        # polygon = Polygon(X, True)
        polygon = Polygon(X)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1.0,edgecolor='none',linewidth=0)

    field = vonMises[:].flatten()
    minField = np.min(field)
    maxField = np.max(field)

    nColors = 128
    color_ind = np.round((nColors-1)*((field-minField)/(maxField-minField)))
    color_ind = color_ind.astype(int)

    maxF = np.max(field)
    colors = 100*field/maxF
    p.set_array(np.array(colors))

    ax2.add_collection(p) 
    ax2.set_title('Displacement & von Mises stress')
    ax2.axis('off')
    ax2.set_xlim([0,L])
    ax2.set_ylim([0,H])
    ax2.axis('equal')

    plt.savefig(fn,dpi=200, bbox_inches='tight', pad_inches=0)

def visu(L,
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
         vonMises,fn):
    fig, (ax1,ax2) = plt.subplots(2, 1)
    fig.patch.set_visible(False)

    diag = np.sqrt((np.max(nodes[:,0])-np.min(nodes[:,0]))**2+(np.max(nodes[:,1])-np.min(nodes[:,1]))**2)
    nodes_deformed = nodes + (scaleDeformation*diag/maxDisp)*displacement

    # plot Mesh & BC
    patches = []
    for i in range(nElem):
        elem =elements[i,:]
        x = nodes[elem,0]
        y = nodes[elem,1]
        X = np.concatenate((np.array([x]).T,np.array([y]).T),axis=1)
        # polygon = Polygon(X, True)
        polygon = Polygon(X)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1.0,edgecolor='black',facecolor='lightgreen',linewidth=1)
    ax1.add_collection(p)

    for i in range(len(dof_load)):
        if (dof_load[i] % 2) == 0:
            ax1.quiver(nodes[int((dof_load[i])/2),0], nodes[int((dof_load[i])/2),1],
                       [np.sign(bglob[dof_load[i],0])],[0],
                       color='red')
        else:
            ax1.quiver(nodes[int((dof_load[i]-1)/2),0], nodes[int((dof_load[i]-1)/2),1],
                       [0],[np.sign( bglob[dof_load[i],0] )],
                       color='red')

    for i in range(len(dof_support)):
        if (dof_support[i] % 2) == 0:
            ax1.plot(nodes[int((dof_support[i])/2),0], nodes[int((dof_support[i])/2),1],
            marker=5,
            markersize=1/scaleDeformation,
            fillstyle='full',
            markerfacecolor='blue',
            markeredgecolor='none')

        else:
            ax1.plot(nodes[int((dof_support[i]-1)/2),0], nodes[int((dof_support[i]-1)/2),1],
            marker=6,
            markersize=1/scaleDeformation,
            fillstyle='full',
            markerfacecolor='blue',
            markeredgecolor='none')

    ax1.set_title('Mesh & BC')
    ax1.axis('off')
    ax1.set_xlim([0,L])
    ax1.set_ylim([0,H])
    ax1.axis('equal')

    # plot Displacement & von Mises stress
    patches = []
    for i in range(nElem):
        elem =elements[i,:]
        x = nodes_deformed[elem,0]
        y = nodes_deformed[elem,1]
        X = np.concatenate((np.array([x]).T,np.array([y]).T),axis=1)
        # polygon = Polygon(X, True)
        polygon = Polygon(X)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1.0,edgecolor='none',linewidth=0)

    field = vonMises[:].flatten()
    minField = np.min(field)
    maxField = np.max(field)

    nColors = 128
    color_ind = np.round((nColors-1)*((field-minField)/(maxField-minField)))
    color_ind = color_ind.astype(int)

    maxF = np.max(field)
    colors = 100*field/maxF
    p.set_array(np.array(colors))

    ax2.add_collection(p)
    ax2.set_title('Displacement & von Mises stress')
    ax2.axis('off')
    ax2.set_xlim([0,L])
    ax2.set_ylim([0,H])
    ax2.axis('equal')

    plt.savefig(fn,dpi=200, bbox_inches='tight', pad_inches=0)