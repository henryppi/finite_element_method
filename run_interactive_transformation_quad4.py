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

vert = transform_quad4(points,vert_init)
points2 = transform_quad4(points,points_init)

vert = deform_grid(vert,
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
    vert = transform_quad4(points,np.copy(vert_init))
    points2 = transform_quad4(points,points_init)
    vert = deform_grid(np.copy(vert),trans_x_val,trans_y_val,rotz_val*np.pi/180,scale_x_val,scale_y_val,shear_x_val,shear_y_val)
    points3 = deform_grid(points,trans_x_val,trans_y_val,rotz_val*np.pi/180,scale_x_val,scale_y_val,shear_x_val,shear_y_val)
    pc.set_verts(vert[elem])
    
    displacement = (points3-points_init).reshape([8,1])
    print(displacement.flatten())
    epsilon,sigma,sigmaVonMises = post_processing(points_init,displacement,E,nu,thickness,order)
    
    line.set_xy(points3)
    vertex.set_xdata(points3[:,0])
    vertex.set_ydata(points3[:,1])

    area = get_area_quad4(points3,order)

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