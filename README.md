[![PyPI - Version](https://img.shields.io/pypi/v/flowreg3d)](https://pypi.org/project/flowreg3d/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/flowreg3d)](https://pypi.org/project/flowreg3d/)
[![PyPI - License](https://img.shields.io/pypi/l/flowreg3d)](LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/flowreg3d)](https://pypistats.org/packages/flowreg3d)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/flowreg3d?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=all+time+downloads)](https://pepy.tech/projects/flowreg3d)
[![GitHub Actions](https://github.com/FlowRegSuite/flowreg3d/actions/workflows/pypi-release.yml/badge.svg)](https://github.com/FlowRegSuite/flowreg3d/actions/workflows/pypi-release.yml)

## 🚧 Under Development

This project is still in an **alpha stage**. Expect rapid changes, incomplete features, and possible breaking updates between releases.

- The API may evolve as we stabilize core functionality.
- Documentation and examples are incomplete.
- Feedback and bug reports are especially valuable at this stage.
- GPU implementation currently produces numerical differences compared to the CPU version and might require different parameter settings.

# <img src="https://raw.githubusercontent.com/flowregsuite/flowreg3D/3df1fddabd74b1f33e3361a8abcb12239eef7f6b/img/flowreglogo.png" alt="FlowReg logo" height="64"> flowreg3D

Python implementation of volumetric optical flow for motion correction in 3D fluorescence microscopy. Building on the 2D Flow-Registration insights, flowreg3D provides **natively 3D dense** motion analysis and correction with **subpixel-precision** for non-rigid motion volumetric microscopy data.

**Related projects**
- Flow-Registration: https://github.com/FlowRegSuite/flow_registration
- PyFlowReg: https://github.com/FlowRegSuite/pyflowreg
- ImageJ/Fiji plugin: https://github.com/FlowRegSuite/flow_registration_IJ
- Napari plugin: https://github.com/FlowRegSuite/napari-flowreg


![Fig1](https://raw.githubusercontent.com/flowregsuite/flowreg3D/3df1fddabd74b1f33e3361a8abcb12239eef7f6b/img/comparison.png)
Alignment of two 2P imaging stacks with difficult, synthetic non-rigid motion patterns.

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


## License

flowreg3D Non-commercial License

Copyright (c) 2025 Okinawa Institute of Science and Technology Graduate University. All rights reserved.

Redistribution and use for non-commercial purposes in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. The software is used solely for non-commercial purposes. It may not be used indirectly for commercial use, including operation on or for a website or service that receives advertising, sponsorship, or other revenue.  For commercial use rights, contact Okinawa Institute of Science and Technology, OIST Innovation, at  tls@oist.jp.

2. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

3. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

4. Neither the name of the Okinawa Institute of Science and Technology Graduate University nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE OKINAWA INSTITUTE OF SCIENCE AND TECHNOLOGY GRADUATE UNIVERSITY AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PATENTS, TRADEMARKS, OR OTHER PROPRIETARY RIGHTS, ARE DISCLAIMED. NO PATENT RIGHTS ARE GRANTED, WHETHER EXPRESSLY, BY IMPLICATION, ESTOPPEL, OR OTHERWISE. IN NO EVENT SHALL THE OKINAWA INSTITUTE OF SCIENCE AND TECHNOLOGY GRADUATE UNIVERSITY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
