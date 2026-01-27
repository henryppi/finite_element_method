import numpy as np
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,Button,TextBox

def shape_fun_quad4_scalar(r,s):
    Nrs =  0.25*np.array([ (1 - r) * (1 - s),\
                  (1 + r) * (1 - s),\
                  (1 + r) * (1 + s), \
                  (1 - r) * (1 + s)])
    return Nrs

def shape_fun_grad_quad4_scalar(r,s):
    dNdX =  0.25*np.array([[-1 + s , 1 - s , 1 + s , -1 - s ],\
                  [-1 + r ,-1 - r , 1 + r ,  1 - r ]])
    return dNdX

def gauss_points_quad(order):
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

def get_area_quad4(points,order):
    gp,gw = gauss_points_quad(order)
    ngp = gp.shape[0]
    area = 0.0
    for ip in range(ngp):
        dNrs = shape_fun_grad_quad4_scalar(gp[ip,0],gp[ip,1])
        J = np.matrix(dNrs)*np.matrix(points)
        detJ = J[0,0]*J[1,1] - J[1,0]*J[0,1]
        area += detJ*gw[ip]
    return area

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

def deform_grid(new_points,
                trans_x,
                trans_y,
                rot_z,
                scale_x,
                scale_y,
                shear_x,
                shear_y):
    
    new_points[:,0] += shear_x*new_points[:,1]
    new_points[:,1] += shear_y*new_points[:,0]

    new_points[:,0] *= scale_x
    new_points[:,1] *= scale_y
    
    R = np.matrix([[np.cos(rot_z),np.sin(rot_z)],[-np.sin(rot_z), np.cos(rot_z)]])
    new_points = (R*new_points.T).T
    
    new_points[:,0] += trans_x
    new_points[:,1] += trans_y
    
    return np.asarray(new_points)


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

def post_processing(nodes,displacement,E,nu,t,order):
    
    D = (E/(1.0-nu**2))*np.array([[ 1.0, nu,  0.0         ],\
                    [ nu,  1.0, 0.0         ],\
                    [ 0.0, 0.0, 0.5*(1.0-nu)]])
    
    epsilon = np.zeros([3,1],float)
    sigma = np.zeros([3,1],float)
    
    gp,gw = gauss_points_quad(order)
    ngp =gp.shape[0]
    
    X = nodes + displacement.reshape([4,2])

    for ip in range(ngp):
        dNrs = shape_fun_grad_quad4_scalar(gp[ip,0],gp[ip,1])
        J = np.matrix(dNrs)*np.matrix(X)
        detJ = J[0,0]*J[1,1] - J[1,0]*J[0,1]
        invJ = (1.0/detJ)*np.matrix([[J[1,1],-J[0,1]],[-J[1,0],J[0,0]]])
        dNdX = invJ*dNrs
        B = np.matrix([[dNdX[0,0], 0.0, dNdX[0,1], 0.0, dNdX[0,2], 0.0, dNdX[0,3], 0.0],\
            [0.0, dNdX[1,0], 0.0, dNdX[1,1], 0.0, dNdX[1,2], 0.0, dNdX[1,3]],\
            [dNdX[1,0], dNdX[0,0], dNdX[1,1], dNdX[0,1], dNdX[1,2], dNdX[0,2], dNdX[1,3], dNdX[0,3]]])

        epsilon += B*displacement*detJ*gw[ip]
        sigma += np.matrix(D)*epsilon
    sigmaVonMises = np.sqrt( sigma[0,0]**2 + sigma[1,0]**2 - sigma[0,0]*sigma[1,0] + 3.0*sigma[2,0]**2 )
    return epsilon.flatten(),sigma.flatten(),sigmaVonMises

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

class gui_control_quad_transform:
    def __init__(self,points):
        self.steps = 20
        self.axcolor = 'lightgoldenrodyellow'

        self.rotz_min = -180.0; self.rotz_val = self.rotz_init = 0.0; self.rotz_max = 180.0
        self.trans_x_min = -2.0; self.trans_x_val = self.trans_x_init = 0.0; self.trans_x_max = 2.0
        self.trans_y_min = -2.0; self.trans_y_val = self.trans_y_init = 0.0; self.trans_y_max = 2.0

        self.scale_x_min = 0.0; self.scale_x_val = self.scale_x_init = 1.0; self.scale_x_max = 2
        self.scale_y_min = 0.0; self.scale_y_val = self.scale_y_init = 1.0; self.scale_y_max = 2

        self.shear_x_min = -1; self.shear_x_val = self.shear_x_init = 0.0; self.shear_x_max = 1
        self.shear_y_min = -1; self.shear_y_val = self.shear_y_init = 0.0; self.shear_y_max = 1

        self.u1x_min = -1.0+points[0,0]; self.u1x_val = self.u1x_init = 0.0+points[0,0]; self.u1x_max = 1.0+points[0,0]
        self.u1y_min = -1.0+points[0,1]; self.u1y_val = self.u1y_init = 0.0+points[0,1]; self.u1y_max = 1.0+points[0,1]
        self.u2x_min = -1.0+points[1,0]; self.u2x_val = self.u2x_init = 0.0+points[1,0]; self.u2x_max = 1.0+points[1,0]
        self.u2y_min = -1.0+points[1,1]; self.u2y_val = self.u2y_init = 0.0+points[1,1]; self.u2y_max = 1.0+points[1,1]
        self.u3x_min = -1.0+points[2,0]; self.u3x_val = self.u3x_init = 0.0+points[2,0]; self.u3x_max = 1.0+points[2,0]
        self.u3y_min = -1.0+points[2,1]; self.u3y_val = self.u3y_init = 0.0+points[2,1]; self.u3y_max = 1.0+points[2,1]
        self.u4x_min = -1.0+points[3,0]; self.u4x_val = self.u4x_init = 0.0+points[3,0]; self.u4x_max = 1.0+points[3,0]
        self.u4y_min = -1.0+points[3,1]; self.u4y_val = self.u4y_init = 0.0+points[3,1]; self.u4y_max = 1.0+points[3,1]

    def get_points(self):
        return  np.array([[self.u1x_init,self.u1y_init],\
                          [self.u2x_init,self.u2y_init],\
                          [self.u3x_init,self.u3y_init],\
                          [self.u4x_init,self.u4y_init]])

    def get_slider_val_transform(self):
        return self.slider_trans_x.val,\
               self.slider_trans_y.val,\
               self.slider_rotz.val,\
               self.slider_scale_x.val,\
               self.slider_scale_y.val,\
               self.slider_shear_x.val,\
               self.slider_shear_y.val

    def get_slider_val_displacement(self):
        return self.slider_u1x.val,self.slider_u1y.val,\
               self.slider_u2x.val,self.slider_u2y.val,\
               self.slider_u3x.val,self.slider_u3y.val,\
               self.slider_u4x.val,self.slider_u4y.val

    def init_slider(self,plt):
        self.ax_rotz = plt.axes([0.6, 0.95, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_trans_x = plt.axes([0.6, 0.90, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_trans_y = plt.axes([0.6, 0.85, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_scale_x = plt.axes([0.6, 0.80, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_scale_y = plt.axes([0.6, 0.75, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_shear_x = plt.axes([0.6, 0.70, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_shear_y = plt.axes([0.6, 0.65, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height

        self.ax_u1x = plt.axes([0.6, 0.40, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_u1y = plt.axes([0.6, 0.35, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_u2x = plt.axes([0.6, 0.30, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_u2y = plt.axes([0.6, 0.25, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_u3x = plt.axes([0.6, 0.20, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_u3y = plt.axes([0.6, 0.15, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_u4x = plt.axes([0.6, 0.10, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height
        self.ax_u4y = plt.axes([0.6, 0.05, 0.3, 0.03], facecolor=self.axcolor) #left bottom width height

        self.slider_rotz = Slider(self.ax_rotz, 'rot z', self.rotz_min, self.rotz_max, valinit=self.rotz_init, valstep=(self.rotz_max-self.rotz_min)/self.steps)
        self.slider_trans_x = Slider(self.ax_trans_x,' trans x', self.trans_x_min, self.trans_x_max, valinit=self.trans_x_init, valstep=(self.trans_x_max-self.trans_x_min)/self.steps)
        self.slider_trans_y = Slider(self.ax_trans_y, 'trans y', self.trans_y_min, self.trans_y_max, valinit=self.trans_y_init, valstep=(self.trans_y_max-self.trans_y_min)/self.steps)
        self.slider_scale_x = Slider(self.ax_scale_x, 'scale x', self.scale_x_min, self.scale_x_max, valinit=self.scale_x_init, valstep=(self.scale_x_max-self.scale_x_min)/self.steps)
        self.slider_scale_y = Slider(self.ax_scale_y, 'scale y', self.scale_y_min, self.scale_y_max, valinit=self.scale_y_init, valstep=(self.scale_y_max-self.scale_y_min)/self.steps)
        self.slider_shear_x = Slider(self.ax_shear_x, 'shear x', self.shear_x_min, self.shear_x_max, valinit=self.shear_x_init, valstep=(self.shear_x_max-self.shear_x_min)/self.steps)
        self.slider_shear_y = Slider(self.ax_shear_y, 'shear y', self.shear_y_min, self.shear_y_max, valinit=self.shear_y_init, valstep=(self.shear_y_max-self.shear_y_min)/self.steps)

        self.slider_u1x = Slider(self.ax_u1x, 'u1x', self.u1x_min, self.u1x_max, valinit=self.u1x_init, valstep=(self.u1x_max-self.u1x_min)/self.steps)
        self.slider_u1y = Slider(self.ax_u1y, 'u1y', self.u1y_min, self.u1y_max, valinit=self.u1y_init, valstep=(self.u1y_max-self.u1y_min)/self.steps)
        self.slider_u2x = Slider(self.ax_u2x, 'u2x', self.u2x_min, self.u2x_max, valinit=self.u2x_init, valstep=(self.u2x_max-self.u2x_min)/self.steps)
        self.slider_u2y = Slider(self.ax_u2y, 'u2y', self.u2y_min, self.u2y_max, valinit=self.u2y_init, valstep=(self.u2y_max-self.u2y_min)/self.steps)
        self.slider_u3x = Slider(self.ax_u3x, 'u3x', self.u3x_min, self.u3x_max, valinit=self.u3x_init, valstep=(self.u3x_max-self.u3x_min)/self.steps)
        self.slider_u3y = Slider(self.ax_u3y, 'u3y', self.u3y_min, self.u3y_max, valinit=self.u3y_init, valstep=(self.u3y_max-self.u3y_min)/self.steps)
        self.slider_u4x = Slider(self.ax_u4x, 'u4x', self.u4x_min, self.u4x_max, valinit=self.u4x_init, valstep=(self.u4x_max-self.u4x_min)/self.steps)
        self.slider_u4y = Slider(self.ax_u4y, 'u4y', self.u4y_min, self.u4y_max, valinit=self.u4y_init, valstep=(self.u4y_max-self.u4y_min)/self.steps)
    
    def observer(self,update_fig):
        self.slider_rotz.on_changed(update_fig)
        self.slider_trans_x.on_changed(update_fig)
        self.slider_trans_y.on_changed(update_fig)
        self.slider_scale_x.on_changed(update_fig)
        self.slider_scale_y.on_changed(update_fig)
        self.slider_shear_x.on_changed(update_fig)
        self.slider_shear_y.on_changed(update_fig)

        self.slider_u1x.on_changed(update_fig)
        self.slider_u1y.on_changed(update_fig)
        self.slider_u2x.on_changed(update_fig)
        self.slider_u2y.on_changed(update_fig)
        self.slider_u3x.on_changed(update_fig)
        self.slider_u3y.on_changed(update_fig)
        self.slider_u4x.on_changed(update_fig)
        self.slider_u4y.on_changed(update_fig)