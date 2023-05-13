import os

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d
import numpy as np
import pickle
import seaborn as sns
sns.set(style='white', font_scale=1.25,
        rc={'text.usetex' : True, 'font.family': 'serif'})


elev, azim = 25, 50
cmap = sns.color_palette('rocket', as_cmap=True)

# data
with open('experiment_single_source.pkl', 'rb') as handle:
    datadict = pickle.load(handle)
points = datadict['points']
normals = datadict['normals']
vert = datadict['vertices']
tri = datadict['faces']
colors = datadict['colors']
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

# search space
plt.close()
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')
s = ax.scatter(*points[ind].T, c=pd[ind], cmap=cmap, s=1, rasterized=True)
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
ax.set(xlabel='x', ylabel='y')
ax.set_axis_off()
ax.view_init(elev, azim)
plt.show()

fig.savefig(__file__.strip('.py') + '.pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=None)
