
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

def plot_3(mesh, input_grid, title = '', fname = 'default', bitmap=False, slicetype = 'middle', colrange=None, colormap=CM_INFERNO , contourplot=False):
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
        
    
    yy, zz = np.meshgrid(mesh.yy, mesh.zz, indexing='ij')
    if contourplot:
        map1 = ax00.contourf(yy, zz, yz_slice, norm=plotnorm, cmap=colormap, nlev=10)
    else:
        map1 = ax00.pcolormesh(yy, zz, yz_slice, norm=plotnorm, cmap=colormap)
    ax00.set_xlabel('y')
    ax00.set_ylabel('z')
    ax00.set_title('%s t = %.4e' % (title, mesh.timestamp) )    
    ax00.set_aspect('equal')
    
    xx, zz = np.meshgrid(mesh.xx, mesh.zz, indexing='ij')
    if contourplot:
        ax10.contourf(xx, zz, xz_slice, norm=plotnorm, cmap=colormap, nlev=10)
    else:
        ax10.pcolormesh(xx, zz, xz_slice, norm=plotnorm, cmap=colormap)
    ax10.set_xlabel('x')
    ax10.set_ylabel('z')
    ax10.set_aspect('equal')
    
    xx, yy = np.meshgrid(mesh.xx, mesh.yy, indexing='ij')
    if contourplot:
        ax11.contourf(xx, yy, xy_slice, norm=plotnorm, cmap=colormap, nlev=10)
    else:
        ax11.pcolormesh(xx, yy, xy_slice, norm=plotnorm, cmap=colormap)
    ax11.set_xlabel('x')
    ax11.set_ylabel('y')
    ax11.set_aspect('equal')
    
    cbar = plt.colorbar(map1, cax=axcbar) 

    if bitmap:
        plt.savefig('%s_%s.png' % (fname, mesh.framenum))
        print('Saved %s_%s.png' % (fname, mesh.framenum))
        plt.close(fig)
         
 
