# Engine-Time-Series-LSTM

## 1. Folder Structure

```
Engine-Time-Series-LSTM
├───data
│   └───turbofan.csv
├───logs
│   ├───log.csv
│   └───train_log.txt
├───notebooks
│   ├───eda.ipynb
│   └───results_analysis.ipynb
├───results
│   └───HP_tuning.csv
├───src
│   ├───inference.py
│   ├───tools.py
│   └───train.py
├───weights
│    └───lstm.h5
├───conda.yml
├───presentation.pdf
└───README.md
```

## 2. Environment setup

- To re-create the development environment, a conda.yml file has been provided in the base folder. 
- Run `conda env create --name <name> --file conda.yml` to create the development environment.

## 3. Training

- Note: The best model has been selected by default. If no parameters are specified, the defaults will be used.
- Training or retraining can be done using the train.py file as follows:

### Training Parameters

```
usage: train.py [-h] [--data_path DATA_PATH] [--log_path LOG_PATH] [--weights_path WEIGHTS_PATH] [--alpha ALPHA] [--sequence_length SEQUENCE_LENGTH]
                [--nodes_per_layer NODES_PER_LAYER] [--dropout DROPOUT] [--activation ACTIVATION] [--batch_size BATCH_SIZE] [--verbose VERBOSE]

options:
  -h, --help                        show this help message and exit
  --data_path DATA_PATH             defaults to "data/turbofan.csv"
  --log_path LOG_PATH               defaults to "log/log.csv"
  --weights_path WEIGHTS_PATH       defaults to "weights/lstm.h5"
  --alpha ALPHA                     defaults to 0.3
  --sequence_length SEQUENCE_LENGTH defaults to 100
  --nodes_per_layer NODES_PER_LAYER defaults to "32:64"
  --dropout DROPOUT                 defaults to 0.4
  --activation ACTIVATION           defaults to "tanh"
  --batch_size BATCH_SIZE           defaults to 128
  --verbose VERBOSE                 defaults to 1
```

### Sample Training Command

- `python -m src.train` (Default parameters)
- `python -m src.train --data_path data/new_turbofan.csv --alpha 0.1 --sequence_length 50 nodes_per_layer 64:64:64` (Custom dataset & parameters)

## 3. Inference

- An inference.py script has been provided as an example to perform future inference and implementation.
- Run `python -m src.inference` to perform a test

## 4. Model Performance

- The baseline linear regression model had achieved an RMSE of 40.48 on the test dataset.
- The tuned LSTM model achieved an RMSE of 16.76; this is a 58.5% improvement. 
- For more information on the use of this model, look at the results_analysis.ipynb file located in the notebooks directory.
- Note: Random weight initialization may cause model to achieve different performance
