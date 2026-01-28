import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PolyCollection

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


u = np.array([0,0,0,0,1,1,0,0]) # displacement vector
vert = np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float)
elem = np.array([[0,1,2,3]])
nVert = vert.shape[0]
order = 2 # gauss integration order

vert_disp = vert + np.reshape(u,[4,2])
nx = 11
ny = 11

xx,yy = np.meshgrid(np.linspace(-1,1,nx),np.linspace(-1,1,ny))
zz = shape_fun_quad4_scalar(xx.flatten(),yy.flatten())

if 1:
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8),subplot_kw={"projection": "3d"})
    surf = ax[0,0].plot_surface(xx,yy,np.reshape(zz[1,:],[nx,ny]), cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax[0,0].plot(vert[:,0],vert[:,1],'.k',markersize=20)
    for i in range(nVert):ax[0,0].text(vert[i,0],vert[i,1],0,s='  '+str(i),color = 'k',fontsize=14)
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel('y')

    surf = ax[0,1].plot_surface(xx,yy,np.reshape(zz[0,:],[nx,ny]), cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax[0,1].plot(vert[:,0],vert[:,1],'.k',markersize=20)
    for i in range(nVert):ax[0,1].text(vert[i,0],vert[i,1],0,s='  '+str(i),color = 'k',fontsize=14)
    ax[0,1].set_xlabel('x')
    ax[0,1].set_ylabel('y')

    surf = ax[1,1].plot_surface(xx,yy,np.reshape(zz[3,:],[nx,ny]), cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax[1,1].plot(vert[:,0],vert[:,1],'.k',markersize=20)
    for i in range(nVert):ax[1,1].text(vert[i,0],vert[i,1],0,s='  '+str(i),color = 'k',fontsize=14)
    ax[1,1].set_xlabel('x')
    ax[1,1].set_ylabel('y')

    surf = ax[1,0].plot_surface(xx,yy,np.reshape(zz[2,:],[nx,ny]), cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax[1,0].plot(vert[:,0],vert[:,1],'.k',markersize=20)
    for i in range(nVert):ax[1,0].text(vert[i,0],vert[i,1],0,s='  '+str(i),color = 'k',fontsize=14)
    ax[1,0].set_xlabel('x')
    ax[1,0].set_ylabel('y')
    plt.savefig('./images/quad4_shape_fun.png',dpi=200)
    plt.show()
    plt.close()

if 1:
    gp,gw = gauss_points_quad(order)
    ngp = gp.shape[0]
    area = 0.0
    for ip in range(ngp):
        dNrs = shape_fun_grad_quad4_scalar(gp[ip,0],gp[ip,1])
        J = np.matrix(dNrs)*np.matrix(vert_disp)
        detJ = J[0,0]*J[1,1] - J[1,0]*J[0,1]
        area += detJ*gw[ip]
    print('area = ',area)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    pc = PolyCollection(vert_disp[elem],\
                        facecolor="#228B22",\
                        # facecolor=None,\
                        edgecolor="#16161D",\
                        alpha=0.5,\
                        linewidth=1)
    ax.add_collection(pc)
    ax.plot(vert_disp[:,0],vert_disp[:,1],linestyle='none',color='k',marker='.',markersize=5)
    for i in range(nVert):
        ax.text(vert_disp[i,0],vert_disp[i,1], '  '+str(i),fontsize=12,color='k')
    plt.savefig('./images/quad4_single_element.png',dpi=200)
    plt.show()
    plt.close()