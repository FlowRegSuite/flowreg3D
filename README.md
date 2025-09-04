[![PyPI - Version](https://img.shields.io/pypi/v/flowreg3d)](https://pypi.org/project/flowreg3d/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/flowreg3d)](https://pypi.org/project/flowreg3d/)
[![PyPI - License](https://img.shields.io/pypi/l/flowreg3d)](LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/flowreg3d)](https://pypistats.org/packages/flowreg3d)
[![GitHub Actions](https://github.com/FlowRegSuite/flowreg3d/actions/workflows/pypi-release.yml/badge.svg)](https://github.com/FlowRegSuite/flowreg3d/actions/workflows/pypi-release.yml)

## 🚧 Under Development

This project is still in an **alpha stage**. Expect rapid changes, incomplete features, and possible breaking updates between releases. 

- The API may evolve as we stabilize core functionality.  
- Documentation and examples are incomplete.  
- Feedback and bug reports are especially valuable at this stage.

# <img src="https://raw.githubusercontent.com/FlowRegSuite/flowreg3d/refs/heads/main/img/flowreglogo.png" alt="FlowReg logo" height="64"> flowreg3D

Python implementation of volumetric optical flow for motion correction in 3D fluorescence microscopy. Building on the 2D Flow-Registration insights, flowreg3D provides **natively dense**, **natively 3D**, and **natively subpixel-precision** motion analysis and correction for volumetric microscopy data.

It builds on the Flow-Registration toolbox for compensation and stabilization of multichannel microscopy videos and extends it to true 3D iamging. 

**Related projects**
- Flow-Registration: https://github.com/FlowRegSuite/flow_registration
- PyFlowReg: https://github.com/FlowRegSuite/pyflowreg
- ImageJ/Fiji plugin: https://github.com/FlowRegSuite/flow_registration_IJ
- Napari plugin: https://github.com/FlowRegSuite/napari-flowreg

![Fig1](https://raw.githubusercontent.com/FlowRegSuite/flowreg3d/refs/heads/main/img/bg.jpg)


## Requirements

This code requires python 3.10 or higher.

Initialize the environment with

```bash
conda create --name flowreg3d python=3.10
conda activate flowreg3d
pip install -r requirements.txt
```

## Installation via pip and conda

```bash
conda create --name flowreg3d python=3.10
conda activate flowreg3d
pip install flowreg3d
```

To install the project with full visualization support, you can install it with the ```vis``` extra:

```bash
pip install flowreg3d[vis]
```

## Getting started

[Examples and notebooks coming soon]

The plugin supports most of the commonly used file types such as HDF5, tiff stacks and matlab mat files. To run the motion compensation, the options need to be defined into a ```OF_options``` object.

## Dataset

The 3D motion benchmark dataset used for our evaluations will be available soon. Meanwhile, synthetic test data with controllable 3D motion fields can be generated using the included `motion_generation` module, which creates biologically-informed displacement patterns including injection/recoil events, rotations, scanning jitter, and other microscopy-specific artifacts.

## Citation

Details on the original method and video results can be found [here](https://www.snnu.uni-saarland.de/flow-registration/).

If you use parts of this code or the plugin for your work, please cite

> "flowreg3D: Volumetric optical flow for motion analysis and correction in 3D fluorescence microscopy," (in preparation), 2025.


and for Flow-Registration

> P. Flotho, S. Nomura, B. Kuhn and D. J. Strauss, "Software for Non-Parametric Image Registration of 2-Photon Imaging Data," J Biophotonics, 2022. [doi:https://doi.org/10.1002/jbio.202100330](https://doi.org/10.1002/jbio.202100330)

BibTeX entry
```
@article{flotea2022a,
    author = {Flotho, P. and Nomura, S. and Kuhn, B. and Strauss, D. J.},
    title = {Software for Non-Parametric Image Registration of 2-Photon Imaging Data},
    year = {2022},
  journal = {J Biophotonics},
  doi = {https://doi.org/10.1002/jbio.202100330}
}
```

