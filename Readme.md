# Air Transformer Multi-task Deep Learning Framework

## Requirements

To use this Transformer deep learning framework, please install the following dependencies:

- Python 
- Pytorch
- xarray
- dask
- netcdf4/zarr
- scikit-learn
- numpy
- pandas

Dependencies can be installed using conda or micromamba, as the following command in linux systems:

```
conda/micromamba install -c conda-forge xarray netcdf4 zarr dask scikit-learn 
conda/micromamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Data
To use the script within this repository, you will need to prepare four types of data files:

- Satellite and other time series feature data: Named Merging_{resolution}_{date}.nc, these files contain time-series data such as satellite observations.
- Static auxiliary data: Named Auxiliary_{resolution}.nc, this includes data like Digital Elevation Models (DEM) which generally do not change over time.
- Annual auxiliary data: Named Annually_{year}_{resolution}.nc, these files contain auxiliary data with an annual time resolution.
- Mask data file: Named ocean.nc, this is used to mask out ocean or other non-relevant areas depending on the study's focus.

All files should be prepared as single files for the temporal scale of the study, as demonstrated in the repository's examples. If the land parameter in Config.py is set to True, areas outside of the non-null grid in ocean.nc will be masked out.

The variables mentioned by the feature_name parameter in the Config.py file need to be present in these three data files. The labels parameter represents the pollutants to be predicted, such as ozone. The non-null values of these variables in the Merging_{resolution}_{date}.nc file will be extracted to train the model.


## Usage

To use the deep learning model framework, follow these steps:

1. Prepare your input dataset with netcdf format, in which the resolution of .nc file is your targeted resolution.
2. Create config file by *Config.py*, please fill in all the paremeters in the file one by one.
3. Calculate the mean and standard deviation of all predictor variables by *Calculating_norm.py*
4. Load training dataset for cells with labels by *Loading_data_training.py*.
5. Train the model using your chosen features, labels, optimizer, loss function, and so no by *Training.py*.
6. Predict full-coverage surface concentration of labels by *Predicting.py*.

```
# Import the model framework
python Config.py

# Define your input data shape and number of classes
python Calculate_norms.py

# Load dataset for training
python Loading_data_training.py

# Train the model
python Training.py

# Predict the surface air pollutant concentrations
python Predicting.py

```