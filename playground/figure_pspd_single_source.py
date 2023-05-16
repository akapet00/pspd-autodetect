import os

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d
import numpy as np
import pickle
import seaborn as sns
sns.set(style='white', font_scale=1.25,
        rc={'text.usetex' : True, 'font.family': 'serif'})


elev, azim = 25, 25
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
ax.view_init(elev, azim)
plt.show()

formats = ['png', 'pdf']
for ext in formats:
    fig.savefig(os.path.join('figures', f'pspd_single_source.{ext}'),
                dpi=350,
                bbox_inches='tight',
                pad_inches=None)