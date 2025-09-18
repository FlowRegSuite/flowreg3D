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
- GPU implementation currently produces numerical differences compared to the CPU version and might require different parameter settings.

# <img src="https://raw.githubusercontent.com/FlowRegSuite/flowreg3d/refs/heads/main/img/flowreglogo.png" alt="FlowReg logo" height="64"> flowreg3D

Python implementation of volumetric optical flow for motion correction in 3D fluorescence microscopy. Building on the 2D Flow-Registration insights, flowreg3D provides **natively 3D dense** motion analysis and correction with **subpixel-precision** for non-rigid motion volumetric microscopy data.

**Related projects**
- Flow-Registration: https://github.com/FlowRegSuite/flow_registration
- PyFlowReg: https://github.com/FlowRegSuite/pyflowreg
- ImageJ/Fiji plugin: https://github.com/FlowRegSuite/flow_registration_IJ
- Napari plugin: https://github.com/FlowRegSuite/napari-flowreg

![Fig1](https://raw.githubusercontent.com/FlowRegSuite/flowreg3d/refs/heads/main/img/bg.jpg)

## Features

- **3D Variational Optical Flow**: Directly estimates dense 3D motion fields between volumetric frames, capturing complex non-rigid deformations with subpixel accuracy.
- **GPU Acceleration**: Optional torch backend with fully GPU-optimized solver for fast processing of large 3D frames.
- **Parallelized Processing**: Efficiently handles long sequences of volumetric data.

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

To install the project with GPU support, you can install it with the ```gpu``` extra:

```bash
pip install flowreg3d[gpu]
```

## Getting started

[Examples and notebooks coming soon]

The plugin supports most of the commonly used file types such as HDF5, tiff stacks and matlab mat files. To run the motion compensation, the options need to be defined into a ```OF_options``` object.

## Dataset

The 3D motion benchmark dataset used for our evaluations will be available for download soon. Meanwhile, synthetic test data with controllable 3D motion fields can be generated using the included `motion_generation` module, which creates biologically-informed displacement patterns including injection/recoil events, rotations, scanning jitter, and other microscopy-specific artifacts.

## Citation

If you use parts of this code or the plugin for your work, please cite

> "flowreg3D: Volumetric optical flow for motion analysis and correction in 3D fluorescence microscopy," (in preparation), 2025.
