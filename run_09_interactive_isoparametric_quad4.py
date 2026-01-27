from sys import platform
if platform == "linux" or platform == "linux2":
    print('running on linux')
    osflag = True
elif platform == "darwin":
    print('running on OSX')
    import matplotlib
    # matplotlib.use('qt4agg')  # Can also use 'tkagg' or 'webagg'
    matplotlib.use('tkagg')  # Can also use 'tkagg' or 'webagg'
#     from matplotlib.patches import Polygon
#     from matplotlib.collections import PatchCollection
    osflag = False
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Slider,Button,TextBox

import copy

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

def make_grid_quad4(width,height,nx,ny):
    xx, yy = np.meshgrid(np.linspace(0,width,nx),np.linspace(0,height,ny))
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

def convert_quad4_to_tri3(vertices,elements):
    elem1 = elements[:,[0,1,2]]
    elem2 = elements[:,[0,2,3]]
    triElem = np.concatenate([elem1,elem2],axis=0)

    return vertices,triElem

def compute_element_neighbors(nodes,elements):
    edge_neighbors = np.zeros(elements.shape,int)
    return edge_neighbors

def convert_quad4_to_quad8(nodes,elements):
    
    return nodes,elements

def convert_tri3_to_tri6(vertices,elements):
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
    edgeElem = edgeInd.reshape(nEdgeVert/3,3)
    edgeElem += nVert
    
    newElem = np.concatenate([elements,edgeElem],axis=1)
    newVert = np.concatenate([vertices,edgeVert],axis=0)
    
    return newVert,newElem

def make_grid(elem_type,width,height,n_div_x,n_div_y):
   
    # quad mesh
    nodes, elements = make_grid_quad4(width,height,n_div_x,n_div_y)
    
    if elem_type=='quad8':
        nodes,elements = convert_quad4_to_quad8(nodes,elements)
    elif elem_type[:3]=='tri':
        nodes,elements = convert_quad4_to_tri3(nodes,elements)
        if elem_type[3]=='6':
            nodes,elements = convert_tri3_to_tri6(nodes,elements)
    else:
        pass
    
    
    return points, elements

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

n_grid_lines = 10
width = 2.0
height = 2.0

points_init = np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float)
points = copy.copy(points_init)

nodes,elements = make_grid_quad4(width,height,n_grid_lines+1,n_grid_lines+1)
nodes = deform_grid(nodes,-0.5*width,-0.5*height,0*np.pi/180,1.0,1.0,0.0,0.0)

nodes_init = copy.deepcopy(nodes)

# nodes,elements = convert_quad4_to_tri3(nodes,elements)

# nodes,elements = convert_tri3_to_tri6(nodes,elements)

nNodes = nodes.shape[0]
nElements = elements.shape[0]

print('#nodes ={}, #elements = {}'.format(nNodes,nElements))

rotz_min = -180.0; rotz_val = rotz_init = 0.0; rotz_max = 180.0
trans_x_min = -2.0; trans_x_val = trans_x_init = 0.0; trans_x_max = 2.0
trans_y_min = -2.0; trans_y_val = trans_y_init = 0.0; trans_y_max = 2.0

scale_x_min = 0.0; scale_x_val = scale_x_init = 1.0; scale_x_max = 2
scale_y_min = 0.0; scale_y_val = scale_y_init = 1.0; scale_y_max = 2

shear_x_min = -1; shear_x_val = shear_x_init = 0.0; shear_x_max = 1
shear_y_min = -1; shear_y_val = shear_y_init = 0.0; shear_y_max = 1

u1x_min = -1.0+points[0,0]; u1x_val = u1x_init = 0.0+points[0,0]; u1x_max = 1.0+points[0,0]
u1y_min = -1.0+points[0,1]; u1y_val = u1y_init = 0.0+points[0,1]; u1y_max = 1.0+points[0,1]
u2x_min = -1.0+points[1,0]; u2x_val = u2x_init = 0.0+points[1,0]; u2x_max = 1.0+points[1,0]
u2y_min = -1.0+points[1,1]; u2y_val = u2y_init = 0.0+points[1,1]; u2y_max = 1.0+points[1,1]
u3x_min = -1.0+points[2,0]; u3x_val = u3x_init = 0.0+points[2,0]; u3x_max = 1.0+points[2,0]
u3y_min = -1.0+points[2,1]; u3y_val = u3y_init = 0.0+points[2,1]; u3y_max = 1.0+points[2,1]
u4x_min = -1.0+points[3,0]; u4x_val = u4x_init = 0.0+points[3,0]; u4x_max = 1.0+points[3,0]
u4y_min = -1.0+points[3,1]; u4y_val = u4y_init = 0.0+points[3,1]; u4y_max = 1.0+points[3,1]

fig = plt.figure(figsize=(12,6))
plt.subplots_adjust(left=0.1,right=0.55, bottom=0.05,top=0.95)
ax = fig.add_subplot(111)

axcolor = 'lightgoldenrodyellow'
ax_rotz = plt.axes([0.6, 0.95, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_trans_x = plt.axes([0.6, 0.90, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_trans_y = plt.axes([0.6, 0.85, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_scale_x = plt.axes([0.6, 0.80, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_scale_y = plt.axes([0.6, 0.75, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_shear_x = plt.axes([0.6, 0.70, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_shear_y = plt.axes([0.6, 0.65, 0.3, 0.03], facecolor=axcolor) #left bottom width height

ax_u1x = plt.axes([0.6, 0.40, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_u1y = plt.axes([0.6, 0.35, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_u2x = plt.axes([0.6, 0.30, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_u2y = plt.axes([0.6, 0.25, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_u3x = plt.axes([0.6, 0.20, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_u3y = plt.axes([0.6, 0.15, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_u4x = plt.axes([0.6, 0.10, 0.3, 0.03], facecolor=axcolor) #left bottom width height
ax_u4y = plt.axes([0.6, 0.05, 0.3, 0.03], facecolor=axcolor) #left bottom width height

steps = 20

slider_rotz = Slider(ax_rotz, 'rot z', rotz_min, rotz_max, valinit=rotz_init, valstep=(rotz_max-rotz_min)/steps)
slider_trans_x = Slider(ax_trans_x,'trans x',trans_x_min,trans_x_max,valinit=trans_x_init,valstep=(trans_x_max-trans_x_min)/steps)
slider_trans_y = Slider(ax_trans_y, 'trans y', trans_y_min, trans_y_max, valinit=trans_y_init, valstep=(trans_y_max-trans_y_min)/steps)
slider_scale_x = Slider(ax_scale_x, 'scale x', scale_x_min, scale_x_max, valinit=scale_x_init, valstep=(scale_x_max-scale_x_min)/steps)
slider_scale_y = Slider(ax_scale_y, 'scale y', scale_y_min, scale_y_max, valinit=scale_y_init, valstep=(scale_y_max-scale_y_min)/steps)
slider_shear_x = Slider(ax_shear_x, 'shear x', shear_x_min, shear_x_max, valinit=shear_x_init, valstep=(shear_x_max-shear_x_min)/steps)
slider_shear_y = Slider(ax_shear_y, 'shear y', shear_y_min, shear_y_max, valinit=shear_y_init, valstep=(shear_y_max-shear_y_min)/steps)

slider_u1x = Slider(ax_u1x, 'u1x', u1x_min, u1x_max, valinit=u1x_init, valstep=(u1x_max-u1x_min)/steps)
slider_u1y = Slider(ax_u1y, 'u1y', u1y_min, u1y_max, valinit=u1y_init, valstep=(u1y_max-u1y_min)/steps)
slider_u2x = Slider(ax_u2x, 'u2x', u2x_min, u2x_max, valinit=u2x_init, valstep=(u2x_max-u2x_min)/steps)
slider_u2y = Slider(ax_u2y, 'u2y', u2y_min, u2y_max, valinit=u2y_init, valstep=(u2y_max-u2y_min)/steps)
slider_u3x = Slider(ax_u3x, 'u3x', u3x_min, u3x_max, valinit=u3x_init, valstep=(u3x_max-u3x_min)/steps)
slider_u3y = Slider(ax_u3y, 'u3y', u3y_min, u3y_max, valinit=u3y_init, valstep=(u3y_max-u3y_min)/steps)
slider_u4x = Slider(ax_u4x, 'u4x', u4x_min, u4x_max, valinit=u4x_init, valstep=(u4x_max-u4x_min)/steps)
slider_u4y = Slider(ax_u4y, 'u4y', u4y_min, u4y_max, valinit=u4y_init, valstep=(u4y_max-u4y_min)/steps)

points = np.array([[u1x_init,u1y_init],[u2x_init,u2y_init],[u3x_init,u3y_init],[u4x_init,u4y_init]])

nodes = transform_quad4(points,nodes_init)
points2 = transform_quad4(points,points_init)

nodes = deform_grid(nodes,
                    trans_x_val,
                    trans_y_val,
                    rotz_val*np.pi/180,
                    scale_x_val,
                    scale_y_val,
                    shear_x_val,
                    shear_y_val)
                    
points3 = deform_grid(points2,
                    trans_x_val,
                    trans_y_val,
                    rotz_val*np.pi/180,
                    scale_x_val,
                    scale_y_val,
                    shear_x_val,
                    shear_y_val)

order = 2
thickness = 0.001
nu = 0.288
E = 206.94e9

displacement = (points3-points_init).reshape([8,1])
epsilon,sigma,sigmaVonMises = post_processing(points_init,displacement,E,nu,thickness,order)
print(epsilon,sigma,sigmaVonMises)
print(points3)
gp,gw = gauss_points_quad(order)
ngp = gp.shape[0]
area = 0.0
for ip in range(ngp):
    dNrs = shape_fun_grad_quad4_scalar(gp[ip,0],gp[ip,1])
    J = np.matrix(dNrs)*np.matrix(points3)
    detJ = J[0,0]*J[1,1] - J[1,0]*J[0,1]
    area += detJ*gw[ip]


# if osflag:
#     pc = PolyCollection(nodes[elements], facecolor="#228B22", alpha=0.5, edgecolor="#16161D", linewidth=0.5)
# else:
#     fac = nodes[elements]
#     nElem = fac.shape[0]
#     patches = []
#     for i in range(nElem):
#         polygon = Polygon(fac[i,:,:], True)
#         patches.append(polygon)
#         
#     pc = PatchCollection(patches, facecolor="#228B22", alpha=0.5, edgecolor="#16161D", linewidth=0.5)

# print(nodes,elements)

pc = PolyCollection(nodes[elements], facecolor="#228B22", alpha=0.5, edgecolor="#16161D", linewidth=0.5)
ax.add_collection(pc)

line, = ax.fill(points3[:,0],points3[:,1], '-k',lw=3, fill=False)
vertex, = ax.plot(points3[:,0],points3[:,1],'ok',lw=3)
# sc = ax.scatter([nodes[:,0]],[nodes[:,1]])

axbox = fig.add_axes([0.6, 0.5, 0.1, 0.05])
text_box = TextBox(axbox, " area")
text_box.set_val("{:.3f}".format(area))

txt1 = ax.text(-3,-2.8,"exx = {:.2e}    sxx = {:.2e}".format(epsilon[0],sigma[0]))
txt2 = ax.text(-3,-3.0,"eyy = {:.2e}    syy = {:.2e}".format(epsilon[1],sigma[1]))
txt3 = ax.text(-3,-3.2,"exy = {:.2e}    sxy = {:.2e}".format(epsilon[2],sigma[2]))

ax.set_aspect('equal')
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])
ax.set_axis_off()

def update_fig(*args):
    trans_x_val = slider_trans_x.val
    trans_y_val = slider_trans_y.val
    rotz_val = slider_rotz.val
    scale_x_val = slider_scale_x.val
    scale_y_val = slider_scale_y.val
    shear_x_val = slider_shear_x.val
    shear_y_val = slider_shear_y.val
    
    u1x = slider_u1x.val; u1y = slider_u1y.val
    u2x = slider_u2x.val; u2y = slider_u2y.val
    u3x = slider_u3x.val; u3y = slider_u3y.val
    u4x = slider_u4x.val; u4y = slider_u4y.val

    points = np.array([[u1x,u1y],[u2x,u2y],[u3x,u3y],[u4x,u4y]])
    nodes = transform_quad4(points,copy.deepcopy(nodes_init))
    points2 = transform_quad4(points,points_init)
    nodes = deform_grid(copy.deepcopy(nodes),trans_x_val,trans_y_val,rotz_val*np.pi/180,scale_x_val,scale_y_val,shear_x_val,shear_y_val)
    points3 = deform_grid(points,trans_x_val,trans_y_val,rotz_val*np.pi/180,scale_x_val,scale_y_val,shear_x_val,shear_y_val)
    pc.set_verts(nodes[elements])
    
    
    displacement = (points3-points_init).reshape([8,1])
    print(displacement.flatten())
    epsilon,sigma,sigmaVonMises = post_processing(points_init,displacement,E,nu,thickness,order)
    
    line.set_xy(points3)
    vertex.set_xdata(points3[:,0])
    vertex.set_ydata(points3[:,1])

    area = 0.0
    for ip in range(ngp):
        dNrs = shape_fun_grad_quad4_scalar(gp[ip,0],gp[ip,1])
        J = np.matrix(dNrs)*np.matrix(points3)
        detJ = J[0,0]*J[1,1] - J[1,0]*J[0,1]
        area += detJ*gw[ip]

    text_box.set_val("{:.3f}".format(area))
    
    txt1.set_text("exx = {:.2e}    sxx = {:.2e}".format(epsilon[0],sigma[0]))
    txt2.set_text("eyy = {:.2e}    syy = {:.2e}".format(epsilon[1],sigma[1]))
    txt3.set_text("exy = {:.2e}    sxy = {:.2e}".format(epsilon[2],sigma[2]))
    
    fig.canvas.draw()
    

slider_rotz.on_changed(update_fig)
slider_trans_x.on_changed(update_fig)
slider_trans_y.on_changed(update_fig)
slider_scale_x.on_changed(update_fig)
slider_scale_y.on_changed(update_fig)
slider_shear_x.on_changed(update_fig)
slider_shear_y.on_changed(update_fig)

slider_u1x.on_changed(update_fig)
slider_u1y.on_changed(update_fig)
slider_u2x.on_changed(update_fig)
slider_u2y.on_changed(update_fig)
slider_u3x.on_changed(update_fig)
slider_u3y.on_changed(update_fig)
slider_u4x.on_changed(update_fig)
slider_u4y.on_changed(update_fig)

plt.show()
