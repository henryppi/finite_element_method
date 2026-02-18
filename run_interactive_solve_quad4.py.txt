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
from my_modules import *


points_init = np.array([[0,0],[1,0],[1,1],[0,1]],float)

gauss_order = 2
thickness = 0.001
nu = 0.288
E = 206.94e9


fig = plt.figure(figsize=(12,6))
plt.subplots_adjust(left=0.1,right=0.55, bottom=0.05,top=0.95)
ax = fig.add_subplot(111)

gctrl = gui_control_quad_solve(points_init)
gctrl.init_slider(plt)

bc_type = np.array([1,1,1,1,1,1,1,1],bool)
force = np.array([0,0,0,0,0,0,0,0],float)
t = 0.001
order = 2
displacement = fem_solve_single_quad(points_init,bc_type,force,E,nu,t,order)
epsilon,sigma,sigmaVonMises = post_processing(points_init,displacement,E,nu,t,order)
sigma_p = compute_principle_stress(sigma)
sigma_p0 = np.copy(sigma_p)
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

gctrl.init_text_field(ax)

ax.set_aspect('equal')
ax.set_xlim([-1,2])
ax.set_ylim([-1,2])
ax.set_axis_off()

def update_fig(*args):
    force = gctrl.get_slider_force()
    radio_vals = gctrl.get_radio_button_values()
    bc_type = np.array([True if x == "fixed" else False for x in radio_vals])
    
    displacement = fem_solve_single_quad(np.copy(points_init),bc_type,1000*force,E,nu,t,order)
    point_disp = displacement.reshape([4,2])
    epsilon,sigma,sigmaVonMises = post_processing(np.copy(points_init),displacement,E,nu,t,order)

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
        sigma_p0 = np.copy(sigma_p)
        sigma_p[0:2] *= 1.0/np.sqrt(np.sum(sigma_p[0:2]**2))

        s11 = np.array([np.sin(sigma_p[2]+np.pi/2)*sigma_p[0],np.cos(sigma_p[2]+np.pi/2)*sigma_p[0]])
        s22 = np.array([np.sin(sigma_p[2])*sigma_p[1],np.cos(sigma_p[2])*sigma_p[1]])
        center = np.mean(points,axis=0)

        quiv_s11.set_offsets(center)
        quiv_s11.set_UVC(s11[0], s11[1], C=None)
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
    gctrl.update_text_field(points,epsilon,sigma,sigma_p0,sigmaVonMises)



gctrl.observer(update_fig)

plt.show()
