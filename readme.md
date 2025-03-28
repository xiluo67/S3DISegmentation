# Projection-Based 3D Point Cloud Segmentation

This project implements a **multi-view projection approach** for 3D point cloud segmentation, combining spherical and Bird's-Eye View (BEV) projections with 2D CNNs for efficient segmentation.


## File Structure and Descriptions

### Core Scripts
- `Train_Test.py`  
  2D CNN training/testing process

### Data Processing
- `Get_image.py`  
  Generates spherical projection images and Bird's-Eye View (BEV) projections from 3D point cloud data. 

- `PP2.py`
  Generates perspective projection images from 3D point cloud data. 

### Models
- `CNN.py`  
   Provides the 2D CNN architecture implementation.

- `testing_group_1-3.py`  
   Implements and tests multiple CNN architectures, with options to extend or scale the models further. 

### Back-projection
- `BEV_backproj.py`  
  Back-projection code for Birds Eye View

- `SP_backproj.py`  
  Back-projection code for sperical projection 

- `PP_backproj.py`  
    Back-projection code for Perspective projection


### Dataset statistics
- `Computs3dis.py`

    This module performs statistical analysis on S3DIS database data.

- `readpts.py`

    This module performs statistical analysis on Factory data.

## Installation

### Create Environment from YAML File

To create a Conda environment using an `env.yaml` file:

```bash
conda env create -f env.yaml -n custom_name
conda activate custom_name
```
