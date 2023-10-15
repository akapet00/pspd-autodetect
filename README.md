# pspd-autodetect

The code for the paper: "Automatic detection of peak spatial-average power density on non-planar skin surface", in proceedings of the 31st International Conference on Software, Telecommunications and Computer Networks (SoftCOM 2023), Split, Croatia.

## Abstract

This paper presents a novel algorithm for the automatic detection of the peak spatially averaged power density on non-planar surfaces of the human body of arbitrary geometry. The algorithm relies only on two input priors as minimal requirements: an unordered set of points on the evaluation surface of interest and the corresponding incident or absorbed power density at each point. To demonstrate its effectiveness, the algorithm is applied to assess the spatially averaged incident power density (IPD) on a realistic model of the human head. Specifically, a computational analysis is conducted considering a theoretical scenario in which the region encompassing the right outer ear is exposed to electromagnetic (EM) energy forming a Gaussian pattern on the corresponding surface. The results are compared to values obtained by using a reference planar evaluation surface prescribed in current international guidelines and standards for limiting human exposure to radio frequency (RF) EM fields. Overall, the findings support the viability of the proposed algorithm as a reliable solution to assess the exposure of anatomical body models in close proximity to wireless devices operating at 6 GHz and above, including millimeter wave (MMW) bands anticipated for the fifth generation (5G) technology standard and future advancements.

## Citation

```bibtex
@inproceedings{Kapetanovic2023Automatic,
    author={Kapetanović, Ante and Poljak, Dragan and Dodig, Hrvoje},
    title={Automatic Detection of Peak Spatial-Average Power Density on Nonplanar Body Models}, 
    year={2023},
    booktitle={2023 International Conference on Software, Telecommunications and Computer Networks (SoftCOM)}, 
    doi={10.23919/SoftCOM58365.2023.10271620}
}
```

## Install

Clone this repository to your local machine:
```bash
git clone https://github.com/akapet00/pspd-autodetect
```

Enter the repository:
```bash
cd pspd-autodetect
```
Install `pspd` preferably within a virtual environment, e.g., by using [Conda](https://www.anaconda.com/download):
```bash
conda create --name pspd python=3.9
pip install --upgrade pip
python -m pip install .
```

## Use

```python
from pspd import PSPD


# load data
points = ...            # required, N-by-3 `numpy.ndarray`
power_density = ...     # required, N-by-1 or N-by-3 `numpy.ndarray`
normals = ...           # optional, N-by-3 `numpy.ndarray`
mesh = ...              # optional, `open3d.geometry.TriangleMesh`

# create the peak-spatial power density instance
pspd = PSPD(points, power_density,
            normals=normals,
            mesh=mesh)

# define the projected area size
area = ...              # the area unit should match the unit of points

# run the search algorithm
pspd.find(area)

# extract the results
res = pspd.get_results()
```

## Reproduce the results

### Experiments

To reproduce the results go to [`playground`](https://github.com/akapet00/pspd-autodetect/tree/main/playground) directory
```bash
cd playground
```

#### Input data

The input data are available in the [`input`](https://github.com/akapet00/pspd-autodetect/tree/main/playground/input) directory. Five Bash scripts are available in it. These scripts contain the commands for running Python files inside the [`python`](https://github.com/akapet00/pspd-autodetect/tree/main/playground/input/python) directory. These files are used in order to obtain the following input data:
* `head.xyz` - original point cloud sampled on the surface of the realistic head model, in meters,
* `head.scaled.xyz` - scaled version of the original point cloud, in centimeters,
* `head.scaled.normals` - estimated surface normals on the scaled point cloud,
* `head.scaled.off` - triangle mesh representing the reconstructed surface,
* `head.scaled.iso.off` - remashed surface and
* `head.scaled.iso.watertight.off` - watertight remashed surface.

#### Executables

To get the gist of how the code actually works, try playing with the code available in the [`tutorial.ipynb`](https://github.com/akapet00/pspd-autodetect/blob/main/tutorial.ipynb) notebook.

Within [`playground`](https://github.com/akapet00/pspd-autodetect/tree/main/playground), there are various Python files available. The main file is [`experiment_single_source.py`](https://github.com/akapet00/pspd-autodetect/blob/main/playground/experiment_single_source.py), where the power density is evaluated in a Gaussian pattern over the head surface. Furthermore, the peak spatial-average power density detection algorithm is used to find the worst-case exposure scenario considering a 4 squared centimeters averaging area (as defined in the ICNIRP guidelines for limiting exposure to electromagnetic fields up to 300 GHz and IEEE standard for safety levels with respect to human exposure to electric, magnetic, and electromagnetic fields). All results are pickled and stored inside the [`output`](https://github.com/akapet00/pspd-autodetect/tree/main/playground/output) directory.

Python files whose name start with `figure_` are used to generate a (part of the) figure provided in the paper.
Running these files is as simple as:
```bash
python figure_*.py
```
Generated figures will be shown and stored in `figures` directory upon closing a pop-up window.

To successfully execute these script, the following Python packages for visualization should be installed manually on the system (recommended version in parentheses):
* Matplotlib (3.7.2)
* PyVista (0.41.1)
* SciPy (1.11.1)
* Seaborn (0.12.2)

## Author

Ante Kapetanović

## License

[MIT](https://github.com/akapet00/pspd-autodetect/blob/main/LICENSE)
