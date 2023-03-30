# Parallelization Suite for SC'23 Artifact

The repo includes the codes, inputs, and scripts to run the experiments. Here follows an explaination for the folders/files.
* All the codes for the experiments are in `codes` directory
* The `indigo_include` contains all the header files
* The `run.sh` script will create `inputs` directory and download 5 inputs to it, it calls the `run_codes.py` script to run all the codes through all the inputs
* The `generate_csv.sh` calls `read_throughput.py` to read the outputs and convert them to csv files
* The Python notebooks that analyze the results and create figures are in `analysis` directory

## How to run the experiments

### Step 1
Download the larger inputs and run all the codes through 5 inputs.
> sh run.sh

### Step 2
Read the throughputs and convert them to csv files.
> sh generate_csv.sh

### Step 3
Update the csv file paths `csv_file = [FILE_PATH]` in the `analysis/analysis_*.ipynb`.

Run the notebook to calculate metrics and generate figures for performance analysis.
