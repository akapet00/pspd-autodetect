import os

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d
import numpy as np
import pickle
from scipy.interpolate import CloughTocher2DInterpolator
import seaborn as sns
sns.set(style='white', font_scale=1.25,
        rc={'text.usetex' : True, 'font.family': 'serif'})


cmap = sns.color_palette('viridis', as_cmap=True)

# data
with open(
    os.path.join('output', 'experiment_single_source.pkl'), 'rb'
    ) as handle:
    datadict = pickle.load(handle)
points = datadict['points']
pd = datadict['power density']
ind = datadict['search space indices']
p = datadict['query point']
nbh = datadict['neighborhood']
area = datadict['surface area']
pspd = datadict['pspd']
cbar_ticks = [np.round(pd.min()),
              np.round(pd.ptp()/2),
              np.round(pd.max())]

# angle of rotation for matplotlib's square patch
idx_at_ymin = np.argmin(nbh[:, 1])
ymin = nbh[idx_at_ymin, 1]
z_at_ymin = nbh[idx_at_ymin, 2]

idx_at_zmin = np.argmin(nbh[:, 2])
zmin = nbh[idx_at_zmin, 2]
y_at_zmin = nbh[idx_at_zmin, 1]

d = z_at_ymin - zmin
r = np.sqrt(np.sqrt(area))
angle_rad = 2 * np.arcsin(d / (2 * r))
angle = np.rad2deg(angle_rad)

# interpolated surface within square patch bounds
Y, Z = np.mgrid[nbh[:, 1].min():nbh[:, 1].max():103j,
                nbh[:, 2].min():nbh[:, 2].max():103j]
Y = Y[1:-1, 1:-1]
Z = Z[1:-1, 1:-1]
xfun = CloughTocher2DInterpolator(nbh[:, 1:], nbh[:, 0])
X = xfun(Y, Z)

# search space
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')
s = ax.scatter(*points[ind].T, c=pd[ind], cmap=cmap, s=0.25, rasterized=True)
cbar = fig.colorbar(s, ax=ax, pad=0, shrink=0.5,
                    label='power density (W/m$^2$)')
cbar.set_ticks(cbar_ticks)
square = Rectangle(xy=(ymin, z_at_ymin),
                   width=np.sqrt(area),
                   height=np.sqrt(area),
                   angle=-angle,
                   ec='w', fc='none')
ax.add_patch(square)
pathpatch_2d_to_3d(square, z=p[0], zdir='x')
ax.set_box_aspect(np.ptp(points[ind], axis=0))
ax.set_axis_off()
ax.view_init(25, 25)

formats = ['png', 'pdf']
for ext in formats:
    fig.savefig(os.path.join('figures', f'pspd_single_source_1.{ext}'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=None)

# hot-spot region, zoom in
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')
ax.scatter(*np.delete(nbh, np.where((nbh==p).all(axis=1))[0], axis=0).T,
           s=7.5, fc='white', ec='k', lw=0.5, rasterized=True)
ax.scatter(*p, s=15, fc='r', ec='k', lw=1, rasterized=True)
square = Rectangle(xy=(ymin, z_at_ymin - 0.15 * z_at_ymin),
                   width=np.sqrt(area),
                   height=np.sqrt(area),
                   angle=-angle,
                   ec='k', fc='none', lw=1.5, zorder=5)
ax.add_patch(square)
pathpatch_2d_to_3d(square, z=nbh[:, 0].max(), zdir='x')
ax.set_box_aspect(np.ptp(nbh, axis=0))
ax.set(xlabel='',
       ylabel='$y$ (mm)',
       zlabel='$z$ (mm)',
       xticks=[],
       yticks=[-4.4, -3.2, -2.0],
       zticks=[-3.1, -1.8, -.5],
       yticklabels=[-44, -32, -20],
       zticklabels=[-31, -18, -5])
ax.xaxis.labelpad = 0
ax.yaxis.labelpad = 15
ax.zaxis.labelpad = 15
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.view_init(5, -15)

formats = ['png', 'pdf']
for ext in formats:
    fig.savefig(os.path.join('figures', f'pspd_single_source_2.{ext}'),
                dpi=300,
                pad_inches=None)

# hot-spot region interpolated, zoom in
elev = 5
azim = -30
ls = LightSource(15, -50)
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, Z,
                       rcount=50,
                       ccount=50,
                       color='lightgray',
                       shade=True,
                       lightsource=ls,
                       linewidth=0,
                       antialiased=False,
                       zorder=1)
ax.plot(*p, 'o', ms=7, mfc='r', mec='k', mew=1.5, zorder=2)
square = Rectangle(xy=(ymin, z_at_ymin - 0.15 * z_at_ymin),
                   width=np.sqrt(area),
                   height=np.sqrt(area),
                   angle=-angle,
                   ec='k', fc='none', lw=2, zorder=3)
ax.add_patch(square)
pathpatch_2d_to_3d(square, z=nbh[:, 0].max(), zdir='x')
ax.set_box_aspect(np.ptp(nbh, axis=0))
ax.set(xlabel='',
       ylabel='',
       zlabel='',
       xticks=[],
       yticks=[-4.4, -3.2, -2.0],
       zticks=[-3.1, -1.8, -.5],
       yticklabels=['', '', ''],
       zticklabels=['', '', ''])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.view_init(elev, azim)
plt.show()

formats = ['png', 'pdf']
for ext in formats:
    fig.savefig(os.path.join('figures', f'pspd_single_source_3.{ext}'),
                dpi=300,
                pad_inches=None)
