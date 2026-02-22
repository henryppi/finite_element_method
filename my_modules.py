import numpy as np
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider,Button,TextBox,RadioButtons
from scipy.sparse import coo_matrix
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def mag(v):
    return np.sqrt(np.sum(v**2))

def rot90(v):
    return np.array([v[1],-v[0]])

def apply_fixed_constraint(K,dof):
    for i in dof:
        K[:,i] = 0.
        K[i,:] = 0.
        K[i,i] = 1.
    return K

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

def shape_fun_quad4_grad(r,s):
    dNdX =  0.25*np.array([[-1 + s , 1 - s , 1 + s , -1 - s ],\
                  [-1 + r ,-1 - r , 1 + r ,  1 - r ]])
    return dNdX

def transform_quad4(points,nodes_init):
    Nrs = shape_fun_quad4(nodes_init[:,0],nodes_init[:,1])
    n = nodes_init.shape[0]
    nodes = np.zeros([n,2],float)
    for i in range(4):
        nodes[:,0] += Nrs[:,i]*points[i,0]
        nodes[:,1] += Nrs[:,i]*points[i,1]
        
    return nodes

def fem_solve_single_quad(nodes,bc_type,force,E,nu,t,order):
    ind_fix = np.where(bc_type==True)[0]
    
    D = (E/(1.0-nu**2))*np.array([[ 1.0, nu,  0.0         ],\
                    [ nu,  1.0, 0.0         ],\
                    [ 0.0, 0.0, 0.5*(1.0-nu)]])
    displacement = np.zeros([8,1],float)

    
    gp,gw = gauss_points_quad(order)
    ngp =gp.shape[0]
    
    X = nodes

    Kloc = np.zeros([8,8],float)
    bloc = np.zeros([8,1],float)
    bloc[:,0] = force[:]
    for ip in range(ngp):
        Nrs = shape_fun_quad4(gp[ip,0],gp[ip,1])
        dNrs = shape_fun_quad4_grad(gp[ip,0],gp[ip,1])
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

def mat2ijv(mat,ind):
    nRow,nCol = mat.shape 
    rows = np.tile(ind,nRow)
    cols = np.repeat(ind,nCol)
    vals = mat.ravel()
    return rows.tolist(),cols.tolist(),vals.tolist()

def compute_principle_stress(sigma):
    ps = np.zeros(3,float)
    ps[1] = 0.5*(sigma[0]-sigma[1])+np.sqrt(0.25*(sigma[0]+sigma[1])**2+sigma[2]**2)
    ps[0] = 0.5*(sigma[0]-sigma[1])-np.sqrt(0.25*(sigma[0]+sigma[1])**2+sigma[2]**2)
    ps[2] = 0.5*np.arctan2(2*sigma[2],(sigma[0]-sigma[1]))#+0.5*np.pi
    return ps

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

def node_select_bbox_2D(vertices,bbox,border=0):
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

def post_processing_global(nodes,elements,u_glob,D):
    gauss_order = 2
    nElem = elements.shape[0]
    epsilon = np.zeros([nElem,3],float)
    sigma = np.zeros([nElem,3],float)
    vonMises = np.zeros([nElem,1],float)

    gp,gw = gauss_points_quad(gauss_order)
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
            Nrs = shape_fun_quad4(gp[ip,0],gp[ip,1])
            dNrs = shape_fun_quad4_grad(gp[ip,0],gp[ip,1])
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

def visu(show_mesh,show_vonMises,show_principal_stress,
         L,
         H,
         nodes,
         elements,
         nElem,
         displacement,
         maxDisp,
         sigma,
         vonMises,
         scaleDeformation,
         ind_load,
         dof_load,
         bglob,
         dof_support,
         fn):
    
    diag = np.sqrt((np.max(nodes[:,0])-np.min(nodes[:,0]))**2+(np.max(nodes[:,1])-np.min(nodes[:,1]))**2)
    nodes_deformed = nodes + (scaleDeformation*diag/maxDisp)*displacement

    if show_mesh:
        fig1, ax1 = plt.subplots(1, 1,figsize=(12,8),facecolor='w',frameon=False)
        fig1.patch.set_visible(False)
        

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

        p = PatchCollection(patches, cmap=cm.jet, alpha=1.0,edgecolor='black',facecolor='lightgreen',linewidth=1)
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


    if show_vonMises:
        fig2, ax2 = plt.subplots(1, 1,figsize=(12,8),facecolor='w',frameon=False)
        fig2.patch.set_visible(False)
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

        p = PatchCollection(patches, cmap=cm.jet, alpha=1.0,edgecolor='none',linewidth=0)

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




    if show_principal_stress:
        fig3, ax3 = plt.subplots(1, 1,figsize=(12,8),facecolor='w',frameon=False)
        fig3.patch.set_visible(False)

        patches = []
        for i in range(nElem):
            elem =elements[i,:]
            x = nodes[elem,0]
            y = nodes[elem,1]
            X = np.concatenate((np.array([x]).T,np.array([y]).T),axis=1)
            # polygon = Polygon(X, True)
            polygon = Polygon(X)
            patches.append(polygon)

        p = PatchCollection(patches, cmap=cm.jet, alpha=1.0,edgecolor='none',linewidth=0)

        field = vonMises[:].flatten()
        minField = np.min(field)
        maxField = np.max(field)

        nColors = 128
        color_ind = np.round((nColors-1)*((field-minField)/(maxField-minField)))
        color_ind = color_ind.astype(int)

        maxF = np.max(field)
        colors = 100*field/maxF
        p.set_array(np.array(colors))

        ax3.add_collection(p)
        ax3.set_title('Displacement & von Mises stress')
        ax3.axis('off')
        ax3.set_xlim([0,L])
        ax3.set_ylim([0,H])
        ax3.axis('equal')


        centers = np.zeros([nElem,2],float)
        sigma_ps = np.zeros([nElem,3],float)
        sigma_ps_mag = np.zeros([nElem,1],float)
        eSizes = np.zeros([nElem,1],float)

        for ie in range(nElem):
            sig = sigma[ie,:]
            if not mag(sig)==0:
                elem = elements[ie,:]
                X = nodes[elem,:]
                centers[ie,:] = np.mean(X,axis=0)
                # cx = np.mean(X[:,0])
                # cy = np.mean(X[:,1])
                eSizes[ie,:] = compute_esize(X)
                sigma_ps[ie,:] = compute_principle_stress(sig)
                sigma_ps_mag[ie] = np.sqrt(np.sum(sigma_ps[ie,0:2]**2))
                
                # print(sigma_p)
                
                # sigma_p0 = np.copy(sigma_ps[ie,:])
        sigma_p_max = np.max(sigma_ps_mag,axis=0)
        for ie in range(nElem):
            sig = sigma[ie,:]
            if not mag(sig)==0:
                sigma_p = sigma_ps[ie,:]
                center = centers[ie,:]
                h = eSizes[ie]
                sigma_p[0:2] *= 1.0/sigma_p_max

                s11 = np.array([np.cos(sigma_p[2]+np.pi/2),np.sin(sigma_p[2]+np.pi/2)])*sigma_p[0]
                s22 = np.array([np.cos(sigma_p[2]),np.sin(sigma_p[2])])*sigma_p[1]

                # s11 *=1/mag(s11) 
                s11 *=h*0.5
                # s22 *=1/mag(s22) 
                s22 *=h*0.5

                ax3.plot([center[0]-s11[0],center[0]+s11[0]],[center[1]-s11[1],center[1]+s11[1]],'--k',lw=2)
                ax3.plot([center[0]-s22[0],center[0]+s22[0]],[center[1]-s22[1],center[1]+s22[1]],'-k',lw=2)

    plt.savefig(fn,dpi=300, bbox_inches='tight', pad_inches=0)

def compute_esize(X):
    return np.sqrt((np.max(X[:,0])-np.min(X[:,0]))**2+(np.max(X[:,1])-np.min(X[:,1]))**2)

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


class gui_control_quad_solve:
    def __init__(self,points_init):
        self.points_init = points_init
        self.steps = 20
        self.axcolor = 'lightgoldenrodyellow'

        self.f1_min = -1; self.f1_val = self.f1_init = 0.0; self.f1_max = 1
        self.f2_min = -1; self.f2_val = self.f2_init = 0.0; self.f2_max = 1
        self.f3_min = -1; self.f3_val = self.f3_init = 0.0; self.f3_max = 1
        self.f4_min = -1; self.f4_val = self.f4_init = 0.0; self.f4_max = 1
        self.f5_min = -1; self.f5_val = self.f5_init = 0.0; self.f5_max = 1
        self.f6_min = -1; self.f6_val = self.f6_init = 0.0; self.f6_max = 1
        self.f7_min = -1; self.f7_val = self.f7_init = 0.0; self.f7_max = 1
        self.f8_min = -1; self.f8_val = self.f8_init = 0.0; self.f8_max = 1

    def get_slider_force(self):
        return np.array([self.slider_f1.val,\
                         self.slider_f2.val,\
                         self.slider_f3.val,\
                         self.slider_f4.val,\
                         self.slider_f5.val,\
                         self.slider_f6.val,\
                         self.slider_f7.val,\
                         self.slider_f8.val])
    
    def get_radio_button_values(self):
        return [self.radio1.value_selected,\
                self.radio2.value_selected,\
                self.radio3.value_selected,\
                self.radio4.value_selected,\
                self.radio5.value_selected,\
                self.radio6.value_selected,\
                self.radio7.value_selected,\
                self.radio8.value_selected]
    
    def init_slider(self,plt):
        self.rax1 = plt.axes([0.6, 0.88, 0.06, 0.11], facecolor=self.axcolor) #left bottom width height
        self.rax2 = plt.axes([0.6, 0.76, 0.06, 0.11], facecolor=self.axcolor) #left bottom width height
        self.rax3 = plt.axes([0.6, 0.64, 0.06, 0.11], facecolor=self.axcolor) #left bottom width height
        self.rax4 = plt.axes([0.6, 0.52, 0.06, 0.11], facecolor=self.axcolor) #left bottom width height
        self.rax5 = plt.axes([0.6, 0.40, 0.06, 0.11], facecolor=self.axcolor) #left bottom width height
        self.rax6 = plt.axes([0.6, 0.28, 0.06, 0.11], facecolor=self.axcolor) #left bottom width height
        self.rax7 = plt.axes([0.6, 0.16, 0.06, 0.11], facecolor=self.axcolor) #left bottom width height
        self.rax8 = plt.axes([0.6, 0.04, 0.06, 0.11], facecolor=self.axcolor) #left bottom width height

        self.radio1 = RadioButtons(self.rax1, ('fixed', 'force'))
        self.radio2 = RadioButtons(self.rax2, ('fixed', 'force'))
        self.radio3 = RadioButtons(self.rax3, ('fixed', 'force'),active=1)
        self.radio4 = RadioButtons(self.rax4, ('fixed', 'force'))
        self.radio5 = RadioButtons(self.rax5, ('fixed', 'force'),active=1)
        self.radio6 = RadioButtons(self.rax6, ('fixed', 'force'),active=1)
        self.radio7 = RadioButtons(self.rax7, ('fixed', 'force'))
        self.radio8 = RadioButtons(self.rax8, ('fixed', 'force'),active=1)

        self.ax_f1 = plt.axes([0.7, 0.92, 0.2, 0.04], facecolor=self.axcolor) #left bottom width height
        self.ax_f2 = plt.axes([0.7, 0.80, 0.2, 0.04], facecolor=self.axcolor) #left bottom width height
        self.ax_f3 = plt.axes([0.7, 0.68, 0.2, 0.04], facecolor=self.axcolor) #left bottom width height
        self.ax_f4 = plt.axes([0.7, 0.56, 0.2, 0.04], facecolor=self.axcolor) #left bottom width height
        self.ax_f5 = plt.axes([0.7, 0.44, 0.2, 0.04], facecolor=self.axcolor) #left bottom width height
        self.ax_f6 = plt.axes([0.7, 0.32, 0.2, 0.04], facecolor=self.axcolor) #left bottom width height
        self.ax_f7 = plt.axes([0.7, 0.20, 0.2, 0.04], facecolor=self.axcolor) #left bottom width height
        self.ax_f8 = plt.axes([0.7, 0.08, 0.2, 0.04], facecolor=self.axcolor) #left bottom width height
        self.slider_f1 = Slider(self.ax_f1, 'P1 Fx', self.f1_min, self.f1_max, valinit=self.f1_init, valstep=(self.f1_max-self.f1_min)/self.steps)
        self.slider_f2 = Slider(self.ax_f2, 'P1 Fy', self.f2_min, self.f2_max, valinit=self.f2_init, valstep=(self.f2_max-self.f2_min)/self.steps)
        self.slider_f3 = Slider(self.ax_f3, 'P2 Fx', self.f3_min, self.f3_max, valinit=self.f3_init, valstep=(self.f3_max-self.f3_min)/self.steps)
        self.slider_f4 = Slider(self.ax_f4, 'P2 Fy', self.f4_min, self.f4_max, valinit=self.f4_init, valstep=(self.f4_max-self.f4_min)/self.steps)
        self.slider_f5 = Slider(self.ax_f5, 'P3 Fx', self.f5_min, self.f5_max, valinit=self.f5_init, valstep=(self.f5_max-self.f5_min)/self.steps)
        self.slider_f6 = Slider(self.ax_f6, 'P3 Fy', self.f6_min, self.f6_max, valinit=self.f6_init, valstep=(self.f6_max-self.f6_min)/self.steps)
        self.slider_f7 = Slider(self.ax_f7, 'P4 Fx', self.f7_min, self.f7_max, valinit=self.f7_init, valstep=(self.f7_max-self.f7_min)/self.steps)
        self.slider_f8 = Slider(self.ax_f8, 'P4 Fy', self.f8_min, self.f8_max, valinit=self.f8_init, valstep=(self.f8_max-self.f8_min)/self.steps)

    def init_text_field(self,ax):
        epsilon = np.zeros(3)
        sigma = np.zeros(3)
        sigma_p0 = np.zeros(3)
        sigmaVonMises = 0.0
        self.txt1 = ax.text(-1,-0.9,"exx = {:.2e}    sxx = {:.2e}".format(epsilon[0],sigma[0]))
        self.txt2 = ax.text(-1,-1.0,"eyy = {:.2e}    syy = {:.2e}".format(epsilon[1],sigma[1]))
        self.txt3 = ax.text(-1,-1.1,"exy = {:.2e}    sxy = {:.2e}".format(epsilon[2],sigma[2]))
        self.txt4 = ax.text(-1,-0.75,"seqv = {:.2e}    s11 = {:.2e}    s22 = {:.2e} theta = {}deg".format(sigmaVonMises,sigma_p0[0],sigma_p0[1],int(sigma_p0[2]*180/np.pi)))

        self.txtp1 = ax.text(self.points_init[0,0]-0.15,self.points_init[0,1]-0.25,"1",fontsize=16)
        self.txtp2 = ax.text(self.points_init[1,0]+0.1,self.points_init[1,1]-0.25,"2",fontsize=16)
        self.txtp3 = ax.text(self.points_init[2,0]+0.1,self.points_init[2,1]+0.15,"3",fontsize=16)
        self.txtp4 = ax.text(self.points_init[3,0]-0.1,self.points_init[3,1]+0.15,"4",fontsize=16)

    def update_text_field(self,points,epsilon,sigma,sigma_p0,sigmaVonMises):
        self.txt1.set_text("exx = {:.2e}    sxx = {:.2e}".format(epsilon[0],sigma[0]))
        self.txt2.set_text("eyy = {:.2e}    syy = {:.2e}".format(epsilon[1],sigma[1]))
        self.txt3.set_text("exy = {:.2e}    sxy = {:.2e}".format(epsilon[2],sigma[2]))
        self.txt4.set_text("seqv = {:.2e}    s11 = {:.2e}    s22 = {:.2e} theta = {}deg".format(sigmaVonMises, sigma_p0[0], sigma_p0[1], int(sigma_p0[2]*180/np.pi)))

        self.txtp1.set_position([points[0,0]-0.15,points[0,1]-0.25])
        self.txtp2.set_position([points[1,0]+0.1,points[1,1]-0.25])
        self.txtp3.set_position([points[2,0]+0.1,points[2,1]+0.15])
        self.txtp4.set_position([points[3,0]-0.1,points[3,1]+0.15])

    def observer(self,update_fig):
        self.slider_f1.on_changed(update_fig)
        self.slider_f2.on_changed(update_fig)
        self.slider_f3.on_changed(update_fig)
        self.slider_f4.on_changed(update_fig)
        self.slider_f5.on_changed(update_fig)
        self.slider_f6.on_changed(update_fig)
        self.slider_f7.on_changed(update_fig)
        self.slider_f8.on_changed(update_fig)

        self.radio1.on_clicked(update_fig)
        self.radio2.on_clicked(update_fig)
        self.radio3.on_clicked(update_fig)
        self.radio4.on_clicked(update_fig)
        self.radio5.on_clicked(update_fig)
        self.radio6.on_clicked(update_fig)
        self.radio7.on_clicked(update_fig)
        self.radio8.on_clicked(update_fig)