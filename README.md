# AI-Powered Cloud Masking Project
In this project, we implemented a complete machine learning pipeline that takes satellite images as input, identifies clouds, and generates a cloud mask as its output.  
This helps in solving the cloud obstruction issue in optical satellite imagery, as identified clouds can be removed, ensuring clearer and more usable images for analysis.

## Requirements

Firstly, you need to install required packages and libraries, it can be found in `requirements.txt`. You can install them using 
```bash
pip install -r requirements.txt
```

Secondly, install dataset from this [Google Drive Link](https://drive.google.com/file/d/1-cU2qx7XY_lwCC7PKOnnNRkeyRto80gC/view), and rename folder to `dataset`.

Make sure to have python 3.10+ installed.

For GPU acceleration, ensure that CUDA-compatible PyTorch is installed.

## Project Structure
```
PROJECT/
├── dataset/                      # Our dataset
├── dataset_filter/               # Text files to filter dataset (eg. noise, clean, missing)
├── evaluation/                   # Model evaluation scripts and metrics
├── models/                       # Saved models
├── thresholds/                   # Thresholds used for our models
│
├── dataset_summary.csv           # Statistical summary of dataset
├── DL.ipynb                      # Building Deep learning model notebook
├── EDA_Preprocessing.ipynb       # Exploratory data analysis and preprocessing
├── ML.ipynb                      # Building Classical model notebook
├── model_logs.txt                # Model size and number of operations
├── profiler.py                   # Code for model profiling
├── Team_6.pdf                    # Project report
├── run_inference.py              # Run inference using a trained model
├── sandbox.ipynb                 # Experimental code and trials
├── ST-Project.pdf                # Project document
├── unet.py                       # U-Net model architecture
├── utilities.py                  # Helper functions
```

## Running The Project

### Test Model

```bash
python3 run_inference.py <test_folder_path> <model_path>
```
Example
```bash
python3 .\run_inference.py test/test/data .\models\instance_norm_e3.pkl
```

