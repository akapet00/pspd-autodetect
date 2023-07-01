# pspd-autodetect

The code for the paper: "Automatic detection of peak spatial-average power density on non-planar skin surface", to be presented at the 31st International Conference on Software, Telecommunications and Computer Networks (SoftCOM 2023) in Split, Croatia.

## Abstract

This paper presents a novel algorithm for the automatic detection of the peak spatially averaged power density on non-planar surfaces of the human body of arbitrary geometry. The algorithm relies only on two input priors as minimal requirements: an unordered set of points on the evaluation surface of interest and the corresponding incident or absorbed power density at each point. To demonstrate its effectiveness, the algorithm is applied to assess the spatially averaged incident power density (IPD) on a realistic model of the human head. Specifically, a computational analysis is conducted considering a theoretical scenario in which the region encompassing the right outer ear is exposed to electromagnetic (EM) energy forming a Gaussian pattern on the corresponding surface. The results are compared to values obtained by using a reference planar evaluation surface prescribed in current international guidelines and standards for limiting human exposure to radio frequency (RF) EM fields. Overall, the findings support the viability of the proposed algorithm as a reliable solution to assess the exposure of anatomical body models in close proximity to wireless devices operating at 6 GHz and above, including millimeter wave (MMW) bands anticipated for the fifth generation (5G) technology standard and future advancements.

## Citation

tba

## Reproduce the results

### Install

Clone this repository to your local machine
```bash
git clone https://github.com/akapet00/pspd-autodetect
```

Enter the repository
```bash
cd pspd-autodetect
```
Install `pspd` by
```bash
pip install .
```

### Use

```python
from pspd import PSPD

# load data
points = ...  # required
power_density = ...  # required
mesh = ...  # optional

# create the peak-spatial power density instance
pspd = PSPD(points, power_density,
            mesh=mesh)

# define the projected area size, unit must correspond to `points`
area = ...

# run the search algorithm
pspd.find(area)

# extract the results
res = pspd.get_results()
```

### Experiments

To reproduce the results go to `playground` directory
```bash
cd playground
```

#### Input data

The input data is available in `input` directory. There, in total 5 .sh scripts are available which hold the command for running .py files within `python` directory. These files are used in order to obtain input data:
* `head.xyz` - originally available point cloud sampled on the surface of the realistic head model in meters
* `head.scaled.xyz` - scaled version of the point cloud in centimeters
* `head.scaled.normals` - estimated surface normals on the scaled point cloud
* `head.scaled.off` - triangulated reconstructed surface 
* `head.scaled.iso.off` - remashed surface
* `head.scaled.iso.watertight.off` - watertight remashed surface

#### Executables

To get the gist on how the code actually works, first try playing with `tutorial.ipynb`.

Within `playground` there are various .py files and 3 directories.

A .py file `experiment_single_source.py` is the main script used for the study. In it, power density is distributed in a Gaussian pattern over the head surface. Additionally, peak-spatial power density detection algorithm is used to find the worst-case exposure scenario considering a 4 squared centimeters averaging area (as defined in the ICNIRP guidelines for limiting exposure to electromagnetic fields up to 300 GHz and IEEE standard for safety levels with respect to human exposure to electric, magnetic, and electromagnetic fields). All results are pickled and stored within `output` directory.

Furthermore, .py files starting with `figure_` were used to generate a (part of the) figure provided in the paper.
Running these files is as simple as
```bash
python figure_*.py
```
Generated figures will be shown and stored in `figures` directory upon closing a matplotlib's pop-up window.


## Author

Ante Kapetanović

## License

[MIT](https://github.com/akapet00/pspd-autodetect/blob/main/LICENSE)
