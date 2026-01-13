import numpy as np
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt

def shape_fun_quad4(r,s):
    r = np.asarray(r).flatten()
    s = np.asarray(s).flatten()
    n = r.shape[0]
    Nrs = np.zeros([n,4],float)
    Nrs[:,0] = 0.25*(1 - r) * (1 - s)
    Nrs[:,1] = 0.25*(1 + r) * (1 - s)
    Nrs[:,2] = 0.25*(1 + r) * (1 + s)
    Nrs[:,3] = 0.25*(1 - r) * (1 + s)
    return Nrs

def transform_quad4(points,nodes_init):
    Nrs = shape_fun_quad4(nodes_init[:,0],nodes_init[:,1])
    n = nodes_init.shape[0]
    nodes = np.zeros([n,2],float)
    for i in range(4):
        nodes[:,0] += Nrs[:,i]*points[i,0]
        nodes[:,1] += Nrs[:,i]*points[i,1]
        
    return nodes

def create_meshgrid_q4(vert0,esize='ndiv',xsize=10,ysize=10):
    if esize=='ndiv':
        nx = int(xsize+1)
        ny = int(ysize+1)
    elif esize=='esize':
        nx = int(ceil(width/(xsize))+1)
        ny = int(ceil(height/(ysize))+1)
    else:
        print('type option: ["ndiv","esize"]')
        return

    vert,elem  = make_grid_quad4(nx,ny)
    
    vert = transform_quad4(vert0,np.copy(vert))
    elem = elem.astype(int)
    return vert,elem

def make_grid_quad4(nx,ny):
    xx, yy = np.meshgrid(np.linspace(-1,1,nx),np.linspace(-1,1,ny))
    nodes = np.zeros([nx*ny,2],float)
    nodes[:,0] = xx.flatten()
    nodes[:,1] = yy.flatten()
    nex = nx-1
    ney = ny-1
    elements = np.zeros([nex*ney,4],int)
    for j in range(ney):
        ind = np.arange(nex)+nex*j
        elements[ind,0] = np.arange(0,nex)  +j*nex+j
        elements[ind,1] = np.arange(1,nex+1) +j*nex+j
        elements[ind,2] = np.arange(1,nex+1) +(j+1)*nex + j+1
        elements[ind,3] = np.arange(0,nex)   +(j+1)*nex + j+1
    
    return nodes,elements

def find_edge_neighbors(elem):    
    nelem,nedge = elem.shape[:]
    neighbor = []
    for i in range(nelem):
        tmp = []
        for j in range(nedge):
            row,col = np.where(elem[i,j]==elem[:,:])
            tmp.extend(row.tolist())

        # vertex neighbor, unused
        tmp2 = np.unique(tmp).tolist()
        tmp2.remove(i)
        
        # edge neighbor
        tmp3 = np.sort(tmp)
        tmp4 = np.diff(tmp3)
        ind = np.where(tmp4==0)[0]
        tmp5 = np.unique(tmp3[ind]).tolist()
        tmp5.remove(i)
        neighbor.append([i,tmp5])

    return neighbor

def get_common_edge(elem1,elem2):
    edge = list(set(elem1).intersection(elem2))
    return edge

def get_internal_external_edges(elem,neighbor):

    internal = []
    external = []
    for ineighbor in neighbor:
        i = ineighbor[0]
        ie = elem[i,:]
        edges = []
        for k in range(ie.shape[0]-1):
            edges.append([ie[k],ie[k+1]])
        edges.append([ie[-1],ie[0]])

        for j in ineighbor[1]:
            edge = get_common_edge(elem[i,:],elem[j,:])
            try:
                edges.index(edge)
            except:
                edge.reverse()
            internal.append(edge)
            edges.remove(edge)

        for ex in edges:
            external.append(ex)

    return np.asarray(internal),np.asarray(external)

def is_point_inside_bbox(point,bbox,tol=0.0):
    return point[0]>=bbox[0]-tol and point[0]<=bbox[1]+tol and point[1]>=bbox[2]-tol and point[1]<=bbox[3]+tol

def select_boundary(vert,elem,edge_ext,bbox,tol=0.0):
    v0 = vert[edge_ext[:,0],:]
    v1 = vert[edge_ext[:,1],:]
    sel = []
    edge_sel = []
    for i in range(edge_ext.shape[0]):
        if is_point_inside_bbox(v0[i,:],bbox,tol) and is_point_inside_bbox(v1[i,:],bbox,tol):
            edge_sel.append(edge_ext[i,:])
            sel.append(i)
    edge_remain = np.delete(edge_ext,sel,axis=0)

    return np.asarray(edge_sel),edge_remain

def plot_mesh(ax,vert,elem,boundary):
    # elements 
    pc = PolyCollection(vert[elem],\
                        facecolor="#228B22",\
                        # facecolor=None,\
                        edgecolor="#16161D",\
                        alpha=0.5,\
                        linewidth=1)
    ax.add_collection(pc)

    # nodes
    ax.plot(vert[:,0],vert[:,1],'.k',markersize=6)

    # boundary edge
    for bnd in boundary:
        edges = bnd[1]
        if bnd[0]=='fixed':
            color = 'b'
        elif bnd[0]=='force':
            color = 'r'
        else:
            color = 'k'
        for e in range(edges.shape[0]):
            edge = edges[e,:]
            ax.plot(vert[edge,0],vert[edge,1],color=color,linestyle='-',lw=2)

    return ax

def plot_mesh_numbers(ax,vert,elem,boundary):
    # node numbers 
    for i in range(vert.shape[0]):
        ax.text(vert[i,0],vert[i,1], ' '+str(i), fontsize = 12, color='k')

    # element numbers
    for j in range(elem.shape[0]):
        ax.text(np.mean(vert[elem[j,:],0]),np.mean(vert[elem[j,:],1]), str(j), fontsize = 12, color='g')
    return ax