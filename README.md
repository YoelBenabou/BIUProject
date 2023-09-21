# Analysis of physiological data obtained from wristband

## Data folder

In this folder you will find two folders: input and WESAD.

- **input folder**: In this folder you have to insert your zip file containing the data you downloaded from the empatica website.
Please be sure the name of the file is input.zip.

- **WESAD folder**: The WESAD dataset.

## Graphs folder

This folder contains some graphs from the training step.

## Model folder

This folder contains all the model we trained in the training part. One of them is the model we used in the prediction step.

## Predictions folder

This folder contains the prediciton graph and predictions txt files for the real data input.

## Installation and Setup

1. **Clone the Repository**

    ```bash
    git clone [repository-url]
    cd [repository-name]
    ```

3. **Run the Script**

    ```bash
    python main_script_name.py
    ```

## Runnable Code files

- **input_data_analysis.py**: It will show you the graphs of each csv file from the input you want to predict.
- **main.py**: The main of the project. There you can train or predict. If you want to predict from input data, please be sure the input.zip file is in the input folder.
- **exploring_WESAD.ipynb**: Some analysis on the WESAD dataset.