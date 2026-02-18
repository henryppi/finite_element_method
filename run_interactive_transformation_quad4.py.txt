from sys import platform
if platform == "linux" or platform == "linux2":
    print('running on linux')
    osflag = True
elif platform == "darwin":
    print('running on OSX')
    import matplotlib
    # matplotlib.use('qt4agg')  
    matplotlib.use('tkagg')  # Can also use 'webagg'
    osflag = False

# modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Slider,Button,TextBox
from my_modules import *

# configuration
width = 2
height = 2
xdiv = 10
ydiv = 10

etype = 'quad4'
esize = 'ndiv'
order = 2
thickness = 0.001
nu = 0.288
E = 206.94e9

steps = 20 #slider steps

unit_square = np.array([[0,0],[1,0],[1,1],[0,1]],float)
points = deform_grid(unit_square,-0.5*width,-0.5*height,0*np.pi/180,width,height,0.0,0.0)
points_init = np.copy(points)

vert,elem = create_meshgrid_q4(points,esize,xdiv,ydiv)
vert_init = np.copy(vert)
nVert = vert.shape[0]
nElem = elem.shape[0]

print('#vert ={}, #elem = {}'.format(nVert,nElem))

gctrl = gui_control_quad_transform(points)

fig = plt.figure(figsize=(12,6))
plt.subplots_adjust(left=0.1,right=0.55, bottom=0.05,top=0.95)
ax = fig.add_subplot(111)

gctrl.init_slider(plt)

points = gctrl.get_points()

vert = transform_quad4(points,vert_init)
points2 = transform_quad4(points,points_init)

vert = deform_grid(vert,
                    gctrl.trans_x_val,
                    gctrl.trans_y_val,
                    gctrl.rotz_val*np.pi/180,
                    gctrl.scale_x_val,
                    gctrl.scale_y_val,
                    gctrl.shear_x_val,
                    gctrl.shear_y_val)
                    
points3 = deform_grid(points2,
                    gctrl.trans_x_val,
                    gctrl.trans_y_val,
                    gctrl.rotz_val*np.pi/180,
                    gctrl.scale_x_val,
                    gctrl.scale_y_val,
                    gctrl.shear_x_val,
                    gctrl.shear_y_val)

displacement = (points3-points_init).reshape([8,1])
epsilon,sigma,sigmaVonMises = post_processing(points_init,displacement,E,nu,thickness,order)
print(epsilon,sigma,sigmaVonMises)
print(points3)

area = get_area_quad4(points3,order)

pc = PolyCollection(vert[elem], facecolor="#228B22", alpha=0.5, edgecolor="#16161D", linewidth=0.5)
ax.add_collection(pc)

line, = ax.fill(points3[:,0],points3[:,1], '-k',lw=3, fill=False)
vertex, = ax.plot(points3[:,0],points3[:,1],'ok',lw=3)

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
    trans_x_val,trans_y_val,rotz_val,scale_x_val,scale_y_val,shear_x_val,shear_y_val\
         = gctrl.get_slider_val_transform()
    u1x,u1y,u2x,u2y,u3x,u3y,u4x,u4y = gctrl.get_slider_val_displacement()

    points = np.array([[u1x,u1y],[u2x,u2y],[u3x,u3y],[u4x,u4y]])
    vert = transform_quad4(points,np.copy(vert_init))
    vert = deform_grid(np.copy(vert),trans_x_val,trans_y_val,rotz_val*np.pi/180,scale_x_val,scale_y_val,shear_x_val,shear_y_val)
    points_tmp = deform_grid(points,trans_x_val,trans_y_val,rotz_val*np.pi/180,scale_x_val,scale_y_val,shear_x_val,shear_y_val)
    pc.set_verts(vert[elem])
    
    displacement = (points_tmp-points_init).reshape([8,1])
    print(displacement.flatten())
    epsilon,sigma,sigmaVonMises = post_processing(points_init,displacement,E,nu,thickness,order)
    
    line.set_xy(points_tmp)
    vertex.set_xdata(points_tmp[:,0])
    vertex.set_ydata(points_tmp[:,1])

    area = get_area_quad4(points_tmp,order)

    text_box.set_val("{:.3f}".format(area))
    txt1.set_text("exx = {:.2e}    sxx = {:.2e}".format(epsilon[0],sigma[0]))
    txt2.set_text("eyy = {:.2e}    syy = {:.2e}".format(epsilon[1],sigma[1]))
    txt3.set_text("exy = {:.2e}    sxy = {:.2e}".format(epsilon[2],sigma[2]))
    
    fig.canvas.draw()

gctrl.observer(update_fig)

plt.show()