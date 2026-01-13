from sys import platform
if platform == "linux" or platform == "linux2":
    print('running on linux')
    osflag = True
elif platform == "darwin":
    print('running on OSX')
    import matplotlib
    matplotlib.use('tkagg')
    osflag = False

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Slider,Button,TextBox,RadioButtons

import copy

def mag(v):
    return np.sqrt(np.sum(v**2))

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

def fem_solve_single_quad(nodes,bc_type,force,E,nu,t,order):
    ind_fix = np.where(bc_type==True)[0]
    
    D = (E/(1.0-nu**2))*np.array([[ 1.0, nu,  0.0         ],\
                    [ nu,  1.0, 0.0         ],\
                    [ 0.0, 0.0, 0.5*(1.0-nu)]])
    displacement = np.zeros([8,1],float)

    
    gp,gw = gaussPointsQuad(order)
    ngp =gp.shape[0]
    
    X = nodes

    Kloc = np.zeros([8,8],float)
    bloc = np.zeros([8,1],float)
    bloc[:,0] = force[:]
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
        Kloc[:,:] += t*B.T*D*B*detJ*gw[ip];
    
    
    Kloc = apply_fixed_constraint(Kloc,ind_fix)
    
    displacement = np.linalg.solve(Kloc, bloc)
    
    return displacement

def post_processing(nodes,displacement,E,nu,t,order):
    
    D = (E/(1.0-nu**2))*np.array([[ 1.0, nu,  0.0         ],\
                    [ nu,  1.0, 0.0         ],\
                    [ 0.0, 0.0, 0.5*(1.0-nu)]])
    
    epsilon = np.zeros([3,1],float)
    sigma = np.zeros([3,1],float)
    
    gp,gw = gaussPointsQuad(order)
    ngp =gp.shape[0]
    
    X = nodes + displacement.reshape([4,2])

    for ip in range(ngp):
        dNrs = shapeFunGradQ4(gp[ip,0],gp[ip,1])
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

def compute_principle_stress(sigma):
    ps = np.zeros(3,float)
    ps[1] = 0.5*(sigma[0]-sigma[1])+np.sqrt(0.25*(sigma[0]+sigma[1])**2+sigma[2]**2)
    ps[0] = 0.5*(sigma[0]-sigma[1])-np.sqrt(0.25*(sigma[0]+sigma[1])**2+sigma[2]**2)
    ps[2] = 0.5*np.arctan2(2*sigma[2],(sigma[0]-sigma[1]))#+0.5*np.pi
    return ps

points_init = np.array([[0,0],[1,0],[1,1],[0,1]],float)

gauss_order = 2
thickness = 0.001
nu = 0.288
E = 206.94e9

fig = plt.figure(figsize=(12,6))
plt.subplots_adjust(left=0.1,right=0.55, bottom=0.05,top=0.95)
ax = fig.add_subplot(111)
axcolor = 'lightgoldenrodyellow'     
rax1 = plt.axes([0.6, 0.88, 0.06, 0.11], facecolor=axcolor) #left bottom width height
rax2 = plt.axes([0.6, 0.76, 0.06, 0.11], facecolor=axcolor) #left bottom width height
rax3 = plt.axes([0.6, 0.64, 0.06, 0.11], facecolor=axcolor) #left bottom width height
rax4 = plt.axes([0.6, 0.52, 0.06, 0.11], facecolor=axcolor) #left bottom width height
rax5 = plt.axes([0.6, 0.40, 0.06, 0.11], facecolor=axcolor) #left bottom width height
rax6 = plt.axes([0.6, 0.28, 0.06, 0.11], facecolor=axcolor) #left bottom width height
rax7 = plt.axes([0.6, 0.16, 0.06, 0.11], facecolor=axcolor) #left bottom width height
rax8 = plt.axes([0.6, 0.04, 0.06, 0.11], facecolor=axcolor) #left bottom width height

radio1 = RadioButtons(rax1, ('fixed', 'force'))
radio2 = RadioButtons(rax2, ('fixed', 'force'))
radio3 = RadioButtons(rax3, ('fixed', 'force'),active=1)
radio4 = RadioButtons(rax4, ('fixed', 'force'))
radio5 = RadioButtons(rax5, ('fixed', 'force'),active=1)
radio6 = RadioButtons(rax6, ('fixed', 'force'),active=1)
radio7 = RadioButtons(rax7, ('fixed', 'force'))
radio8 = RadioButtons(rax8, ('fixed', 'force'),active=1)

# for circle in radio1.circles:circle.set_radius(0.08)
# for circle in radio2.circles:circle.set_radius(0.08)
# for circle in radio3.circles:circle.set_radius(0.08)
# for circle in radio4.circles:circle.set_radius(0.08)
# for circle in radio5.circles:circle.set_radius(0.08)
# for circle in radio6.circles:circle.set_radius(0.08)
# for circle in radio7.circles:circle.set_radius(0.08)
# for circle in radio8.circles:circle.set_radius(0.08)

steps=20
f1_min = -1; f1_val = f1_init = 0.0; f1_max = 1
f2_min = -1; f2_val = f2_init = 0.0; f2_max = 1
f3_min = -1; f3_val = f3_init = 0.0; f3_max = 1
f4_min = -1; f4_val = f4_init = 0.0; f4_max = 1
f5_min = -1; f5_val = f5_init = 0.0; f5_max = 1
f6_min = -1; f6_val = f6_init = 0.0; f6_max = 1
f7_min = -1; f7_val = f7_init = 0.0; f7_max = 1
f8_min = -1; f8_val = f8_init = 0.0; f8_max = 1

ax_f1 = plt.axes([0.7, 0.92, 0.2, 0.04], facecolor=axcolor) #left bottom width height
ax_f2 = plt.axes([0.7, 0.80, 0.2, 0.04], facecolor=axcolor) #left bottom width height
ax_f3 = plt.axes([0.7, 0.68, 0.2, 0.04], facecolor=axcolor) #left bottom width height
ax_f4 = plt.axes([0.7, 0.56, 0.2, 0.04], facecolor=axcolor) #left bottom width height
ax_f5 = plt.axes([0.7, 0.44, 0.2, 0.04], facecolor=axcolor) #left bottom width height
ax_f6 = plt.axes([0.7, 0.32, 0.2, 0.04], facecolor=axcolor) #left bottom width height
ax_f7 = plt.axes([0.7, 0.20, 0.2, 0.04], facecolor=axcolor) #left bottom width height
ax_f8 = plt.axes([0.7, 0.08, 0.2, 0.04], facecolor=axcolor) #left bottom width height
slider_f1 = Slider(ax_f1, 'P1 Fx', f1_min, f1_max, valinit=f1_init, valstep=(f1_max-f1_min)/steps)
slider_f2 = Slider(ax_f2, 'P1 Fy', f2_min, f2_max, valinit=f2_init, valstep=(f2_max-f2_min)/steps)
slider_f3 = Slider(ax_f3, 'P2 Fx', f3_min, f3_max, valinit=f3_init, valstep=(f3_max-f3_min)/steps)
slider_f4 = Slider(ax_f4, 'P2 Fy', f4_min, f4_max, valinit=f4_init, valstep=(f4_max-f4_min)/steps)
slider_f5 = Slider(ax_f5, 'P3 Fx', f5_min, f5_max, valinit=f5_init, valstep=(f5_max-f5_min)/steps)
slider_f6 = Slider(ax_f6, 'P3 Fy', f6_min, f6_max, valinit=f6_init, valstep=(f6_max-f6_min)/steps)
slider_f7 = Slider(ax_f7, 'P4 Fx', f7_min, f7_max, valinit=f7_init, valstep=(f7_max-f7_min)/steps)
slider_f8 = Slider(ax_f8, 'P4 Fy', f8_min, f8_max, valinit=f8_init, valstep=(f8_max-f8_min)/steps)

bc_type = np.array([1,1,1,1,1,1,1,1],bool)
force = np.array([0,0,0,0,0,0,0,0],float)
t = 0.001
order = 2
displacement = fem_solve_single_quad(points_init,bc_type,force,E,nu,t,order)
epsilon,sigma,sigmaVonMises = post_processing(points_init,displacement,E,nu,t,order)
sigma_p = compute_principle_stress(sigma)
sigma_p0 = copy.copy(sigma_p)
sigma_p[0:2] *= 1.0/np.sqrt(np.sum(sigma_p[0:2]**2))
s11 = np.array([np.sin(sigma_p[2]+np.pi/2)*sigma_p[0],np.cos(sigma_p[2]+np.pi/2)*sigma_p[0]])
s22 = np.array([np.sin(sigma_p[2])*sigma_p[1],np.cos(sigma_p[2])*sigma_p[1]])
center = np.mean(points_init,axis=0)

quiv_s11 = ax.quiver(center[0], center[1], s11[0], s11[1], color='magenta',scale_units='xy',scale=1.5,linewidths=4)
quiv_s22 = ax.quiver(center[0], center[1], s22[0], s22[1], color='cyan',scale_units='xy',scale=1.5,linewidths=4)

# print(epsilon,sigma,sigmaVonMises
bc_marker = [5,6,4,6,4,7,5,7]
dofl = [0,0,1,1,2,2,3,3]
sc_list = []
for i in range(len(dofl)):
    sc_obj = ax.scatter([points_init[dofl[i],0]],[points_init[dofl[i],1]],s=500,c='blue', marker=bc_marker[i])
    sc_list.append(sc_obj)
fx = np.zeros(4)
fy = np.zeros(4)

line, = ax.fill(points_init[:,0],points_init[:,1], '-',lw=2, fill=True,facecolor="#228B22",alpha=0.5,edgecolor="#16161D")
vertex, = ax.plot(points_init[:,0],points_init[:,1],'ok',lw=3)

quiv = ax.quiver(points_init[:,0], points_init[:,1], fx, fy, color='red',scale_units='xy',scale=1.0,linewidths=4)

txt1 = ax.text(-1,-0.9,"exx = {:.2e}    sxx = {:.2e}".format(epsilon[0],sigma[0]))
txt2 = ax.text(-1,-1.0,"eyy = {:.2e}    syy = {:.2e}".format(epsilon[1],sigma[1]))
txt3 = ax.text(-1,-1.1,"exy = {:.2e}    sxy = {:.2e}".format(epsilon[2],sigma[2]))
txt4 = ax.text(-1,-0.75,"seqv = {:.2e}    s11 = {:.2e}    s22 = {:.2e} theta = {}deg".format(sigmaVonMises,sigma_p0[0],sigma_p0[1],int(sigma_p0[2]*180/np.pi)))

txtp1 = ax.text(points_init[0,0]-0.15,points_init[0,1]-0.25,"1",fontsize=16)
txtp2 = ax.text(points_init[1,0]+0.1,points_init[1,1]-0.25,"2",fontsize=16)
txtp3 = ax.text(points_init[2,0]+0.1,points_init[2,1]+0.15,"3",fontsize=16)
txtp4 = ax.text(points_init[3,0]-0.1,points_init[3,1]+0.15,"4",fontsize=16)

ax.set_aspect('equal')
ax.set_xlim([-1,2])
ax.set_ylim([-1,2])
ax.set_axis_off()

def update_fig(*args):
    force = np.array([slider_f1.val,slider_f2.val,slider_f3.val,slider_f4.val,slider_f5.val,slider_f6.val,slider_f7.val,slider_f8.val])
    
    radio_vals = [radio1.value_selected,radio2.value_selected,radio3.value_selected,radio4.value_selected,radio5.value_selected,radio6.value_selected,radio7.value_selected,radio8.value_selected]
    bc_type = np.array([True if x == "fixed" else False for x in radio_vals])
    
    displacement = fem_solve_single_quad(copy.copy(points_init),bc_type,1000*force,E,nu,t,order)
    point_disp = displacement.reshape([4,2])
    epsilon,sigma,sigmaVonMises = post_processing(copy.copy(points_init),displacement,E,nu,t,order)

    points = points_init + 1e5*point_disp
    
    line.set_xy(points)
    vertex.set_xdata(points[:,0])
    vertex.set_ydata(points[:,1])
    
    
    quiv.set_offsets(points)
    fx = force[[0,2,4,6]]
    fy = force[[1,3,5,7]]
    quiv.set_UVC(fx, fy, C=None)

    sigma_p0 = np.zeros(3,float)
    if not mag(sigma)==0:
        
        sigma_p = compute_principle_stress(sigma)
        sigma_p0 = copy.copy(sigma_p)
        sigma_p[0:2] *= 1.0/np.sqrt(np.sum(sigma_p[0:2]**2))

        
        s11 = np.array([np.sin(sigma_p[2]+np.pi/2)*sigma_p[0],np.cos(sigma_p[2]+np.pi/2)*sigma_p[0]])
        s22 = np.array([np.sin(sigma_p[2])*sigma_p[1],np.cos(sigma_p[2])*sigma_p[1]])
        center = np.mean(points,axis=0)

        # print(mag(s11),mag(s22))
        # if not mag(s11)==np.nan:# zero quiver length creates plot disruption
        quiv_s11.set_offsets(center)
        quiv_s11.set_UVC(s11[0], s11[1], C=None)
        # if not mag(s22)==np.nan:
        quiv_s22.set_offsets(center)
        quiv_s22.set_UVC(s22[0], s22[1], C=None)
    else:
        quiv_s11.set_UVC(0, 0, C=None)
        quiv_s22.set_UVC(0, 0, C=None)
        
    
    for i in range(len(dofl)):
        sc_list[i].set_offsets(points[dofl[i],:])
        if not bc_type[i]:
            sc_list[i].set_sizes([0])
        else:
            sc_list[i].set_sizes([500])
            
    txt1.set_text("exx = {:.2e}    sxx = {:.2e}".format(epsilon[0],sigma[0]))
    txt2.set_text("eyy = {:.2e}    syy = {:.2e}".format(epsilon[1],sigma[1]))
    txt3.set_text("exy = {:.2e}    sxy = {:.2e}".format(epsilon[2],sigma[2]))
    txt4.set_text("seqv = {:.2e}    s11 = {:.2e}    s22 = {:.2e} theta = {}deg".format(sigmaVonMises, sigma_p0[0], sigma_p0[1], int(sigma_p0[2]*180/np.pi)))

    txtp1.set_position([points[0,0]-0.15,points[0,1]-0.25])
    txtp2.set_position([points[1,0]+0.1,points[1,1]-0.25])
    txtp3.set_position([points[2,0]+0.1,points[2,1]+0.15])
    txtp4.set_position([points[3,0]-0.1,points[3,1]+0.15])
    
    
    
slider_f1.on_changed(update_fig)
slider_f2.on_changed(update_fig)
slider_f3.on_changed(update_fig)
slider_f4.on_changed(update_fig)
slider_f5.on_changed(update_fig)
slider_f6.on_changed(update_fig)
slider_f7.on_changed(update_fig)
slider_f8.on_changed(update_fig)

radio1.on_clicked(update_fig)
radio2.on_clicked(update_fig)
radio3.on_clicked(update_fig)
radio4.on_clicked(update_fig)
radio5.on_clicked(update_fig)
radio6.on_clicked(update_fig)
radio7.on_clicked(update_fig)
radio8.on_clicked(update_fig)
plt.show()
