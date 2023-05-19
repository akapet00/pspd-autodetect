import os

import matplotlib.pyplot as plt
import numpy as np
from pspd.misc import edblquad
import seaborn as sns
sns.set(style='white', font_scale=1.5,
        rc={'text.usetex' : True, 'font.family': 'serif'})

from single_source import generate_power_density


# constants
AMPLITUDE = 10  # W/m2
RADIUS = 2.5   # cm
QUERY_POINT = np.array([0, 0, 0])  # cm
PROJECTED_AREA = 4  # cm2
SCALER = [1, 0.5, 1]

# data
a = np.sqrt(PROJECTED_AREA)
Y, X = np.mgrid[-a/2:a/2:101j, -a/2:a/2:101j]
extent = [X.min(), X.max(), Y.min(), Y.max()]
points = np.c_[X.ravel(), Y.ravel(), np.zeros((X.size, ))]

# generate power density
power_density = generate_power_density(AMPLITUDE,
                                       RADIUS,
                                       QUERY_POINT,
                                       points,
                                       SCALER)

# spatially averaged power density
spd = edblquad(points[:, :2],
               power_density,
               bbox=extent,
               method='gauss') / PROJECTED_AREA
print(spd)

# evaluation plane
fig = plt.figure()
ax = plt.axes()
c = ax.imshow(power_density.reshape(101, 101),
              cmap='viridis',
              aspect='equal',
              interpolation='gaussian',
              origin='lower',
              extent=extent)
ax.scatter(QUERY_POINT[0], QUERY_POINT[1], s=50, c='r', ec='k', lw=1)
cbar = fig.colorbar(c, ax=ax, label='power density (W/m$^2$)')
cbar.set_ticks([power_density.min(),
                (power_density.min()+AMPLITUDE)/2,
                AMPLITUDE])
cbar.set_ticklabels([round(power_density.min()),
                     round((power_density.min()+AMPLITUDE)/2),
                     AMPLITUDE])
ax.set(xlabel='$x$ (cm)',
       ylabel='$y$ (cm)',
       xticks=[round(-a/2), 0, round(a/2)],
       xticklabels=[round(-a/2), 0, round(a/2)],
       yticks=[round(-a/2), 0, round(a/2)],
       yticklabels=[round(-a/2), 0, round(a/2)],)
plt.show()

formats = ['png', 'pdf']
for ext in formats:
    fig.savefig(os.path.join('figures', f'evaluation_plane.{ext}'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=None)
