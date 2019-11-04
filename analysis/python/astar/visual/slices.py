
'''
    Copyright (C) 2014-2019, Johannes Pekkilae, Miikka Vaeisalae.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
'''
import pylab as plt 
import numpy as np 
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

CM_INFERNO = plt.get_cmap('inferno')

def plot_3(mesh, input_grid, title = '', fname = 'default', bitmap=False,
	   slicetype = 'middle', colrange=None, colormap=CM_INFERNO ,
           contourplot=False, points_from_centre = -1, bfieldlines=False, velfieldlines=False):
    fig = plt.figure(figsize=(8, 8))
    grid = gridspec.GridSpec(2, 3, wspace=0.4, hspace=0.4, width_ratios=[1,1, 0.15])
    ax00   = fig.add_subplot( grid[0,0] )
    ax10   = fig.add_subplot( grid[0,1] )
    ax11   = fig.add_subplot( grid[1,1] )
    axcbar = fig.add_subplot( grid[:,2] )

    print(mesh.minfo.contents.keys())

    if slicetype == 'middle':
        yz_slice = input_grid[mesh.xmid, :, :]
        xz_slice = input_grid[:, mesh.ymid, :]
        xy_slice = input_grid[:, :, mesh.zmid]
        if colrange==None:
            plotnorm = colors.Normalize(vmin=input_grid.min(),vmax=input_grid.max()) 
        else:
            plotnorm = colors.Normalize(vmin=colrange[0],vmax=colrange[1]) 
    elif slicetype == 'sum':
        yz_slice = np.sum(input_grid, axis=0) 
        xz_slice = np.sum(input_grid, axis=1) 
        xy_slice = np.sum(input_grid, axis=2) 
        cmin = np.amin([yz_slice.min(), xz_slice.min(), xy_slice.min()])
        cmax = np.amax([yz_slice.max(), xz_slice.max(), xy_slice.max()])
        if colrange==None:
            plotnorm = colors.Normalize(vmin=cmin,vmax=cmax) 
        else:
            plotnorm = colors.Normalize(vmin=colrange[0],vmax=colrange[1]) 
       
    if points_from_centre > 0:
        yz_slice = yz_slice[int(yz_slice.shape[0]/2)-points_from_centre : int(yz_slice.shape[0]/2)+points_from_centre,
                            int(yz_slice.shape[1]/2)-points_from_centre : int(yz_slice.shape[1]/2)+points_from_centre]
        xz_slice = xz_slice[int(xz_slice.shape[0]/2)-points_from_centre : int(xz_slice.shape[0]/2)+points_from_centre,
                            int(xz_slice.shape[1]/2)-points_from_centre : int(xz_slice.shape[1]/2)+points_from_centre]
        xy_slice = xy_slice[int(xy_slice.shape[0]/2)-points_from_centre : int(xy_slice.shape[0]/2)+points_from_centre,
                            int(xy_slice.shape[1]/2)-points_from_centre : int(xy_slice.shape[1]/2)+points_from_centre]
        mesh.xx = mesh.xx[int(mesh.xx.shape[0]/2)-points_from_centre : int(mesh.xx.shape[0]/2)+points_from_centre] 
        mesh.yy = mesh.yy[int(mesh.yy.shape[0]/2)-points_from_centre : int(mesh.yy.shape[0]/2)+points_from_centre]  
        mesh.zz = mesh.zz[int(mesh.zz.shape[0]/2)-points_from_centre : int(mesh.zz.shape[0]/2)+points_from_centre] 
    
    yy, zz = np.meshgrid(mesh.yy, mesh.zz, indexing='ij')
    if contourplot:
        map1 = ax00.contourf(yy, zz, yz_slice, norm=plotnorm, cmap=colormap, nlev=10)
    else:
        map1 = ax00.pcolormesh(yy, zz, yz_slice, norm=plotnorm, cmap=colormap)
    ax00.set_xlabel('y')
    ax00.set_ylabel('z')
    ax00.set_title('%s t = %.4e' % (title, mesh.timestamp) )    
    ax00.set_aspect('equal')

    ax00.contour(yy, zz, np.sqrt((yy-yy.max()/2.0)**2.0 + (zz-zz.max()/2.0)**2.0), [mesh.minfo.contents["AC_accretion_range"]]) 
    
    xx, zz = np.meshgrid(mesh.xx, mesh.zz, indexing='ij')
    if contourplot:
        ax10.contourf(xx, zz, xz_slice, norm=plotnorm, cmap=colormap, nlev=10)
    else:
        ax10.pcolormesh(xx, zz, xz_slice, norm=plotnorm, cmap=colormap)
    ax10.set_xlabel('x')
    ax10.set_ylabel('z')
    ax10.set_aspect('equal')

    ax10.contour(xx, zz, np.sqrt((xx-xx.max()/2.0)**2.0 + (zz-zz.max()/2.0)**2.0), [mesh.minfo.contents["AC_accretion_range"]]) 
    
    xx, yy = np.meshgrid(mesh.xx, mesh.yy, indexing='ij')
    if contourplot:
        ax11.contourf(xx, yy, xy_slice, norm=plotnorm, cmap=colormap, nlev=10)
    else:
        ax11.pcolormesh(xx, yy, xy_slice, norm=plotnorm, cmap=colormap)
    ax11.set_xlabel('x')
    ax11.set_ylabel('y')
    ax11.set_aspect('equal')

    ax11.contour(xx, yy, np.sqrt((xx-xx.max()/2.0)**2.0 + (yy-yy.max()/2.0)**2.0), [mesh.minfo.contents["AC_accretion_range"]]) 

    if bfieldlines:
        ax00.streamplot(mesh.yy, mesh.zz, np.mean(mesh.bb[1], axis=0), np.mean(mesh.bb[2], axis=0))
        ax10.streamplot(mesh.xx, mesh.zz, np.mean(mesh.bb[0], axis=1), np.mean(mesh.bb[2], axis=1))
        ax11.streamplot(mesh.xx, mesh.yy, np.mean(mesh.bb[0], axis=2), np.mean(mesh.bb[1], axis=2))

        #ax00.streamplot(mesh.yy, mesh.zz, mesh.bb[1][mesh.xmid, :, :], mesh.bb[2][mesh.xmid, :, :])
        #ax10.streamplot(mesh.xx, mesh.zz, mesh.bb[0][:, mesh.ymid, :], mesh.bb[2][:, mesh.ymid, :])
        #ax11.streamplot(mesh.xx, mesh.yy, mesh.bb[0][:, : ,mesh.zmid], mesh.bb[1][:, :, mesh.zmid])

        #ax00.quiver(mesh.bb[2][mesh.xmid, ::10, ::10], mesh.bb[1][mesh.xmid, ::10, ::10], pivot='middle')
        #ax10.quiver(mesh.bb[2][::10, mesh.ymid, ::10], mesh.bb[0][::10, mesh.ymid, ::10], pivot='middle')
        #ax11.quiver(mesh.bb[1][::10, ::10, mesh.zmid], mesh.bb[0][::10, ::10, mesh.zmid], pivot='middle')

        #ax00.quiver(mesh.yy, mesh.zz, mesh.bb[2][mesh.xmid, :, :], mesh.bb[1][mesh.xmid, :, :], pivot='middle')
        #ax10.quiver(mesh.xx, mesh.zz, mesh.bb[2][:, mesh.ymid, :], mesh.bb[0][:, mesh.ymid, :], pivot='middle')
        #ax11.quiver(mesh.xx, mesh.yy, mesh.bb[1][:, :, mesh.zmid], mesh.bb[0][:, :, mesh.zmid], pivot='middle')

    if velfieldlines:
        ax00.streamplot(mesh.yy, mesh.zz, mesh.uu[2][mesh.xmid, :, :], mesh.uu[1][mesh.xmid, :, :])
        ax10.streamplot(mesh.xx, mesh.zz, mesh.uu[2][:, mesh.ymid, :], mesh.uu[0][:, mesh.ymid, :])
        ax11.streamplot(mesh.xx, mesh.yy, mesh.uu[1][:, :, mesh.zmid], mesh.uu[0][:, : ,mesh.zmid])
    
    cbar = plt.colorbar(map1, cax=axcbar) 

    if bitmap:
        plt.savefig('%s_%s.png' % (fname, mesh.framenum))
        print('Saved %s_%s.png' % (fname, mesh.framenum))
        plt.close(fig)
         
 
