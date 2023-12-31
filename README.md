# Parallelization Suite for SC'23 Artifact

The repo includes the codes, inputs, and scripts to run the experiments. Here follows an explaination for the folders/files.
* All the codes for the experiments are in `codes` directory
* The Jupyter Notebooks that analyze the results and create figures are in `analysis` directory
* The `data` contains the data from our experiment, running Jupyter Notebooks through the data can generate the expected figures in the paper
* The `include` contains all the header files
* The `scripts` contains all the Python scripts that help to run the tests and read the throughputs
* The `run_all.sh` script will create `inputs` directory and download 5 inputs to it, it calls the `scripts/run_all_codes.py` script to run all the codes through all the inputs
* The `run_cpp.sh` run C++ codes for partial experiment
* The `generate_*_csv.sh` calls `scripts/read_throughput.py` to read the outputs and convert them to csv files

## Software
We used GCC 11.3.1 compiler for OpenMP and C++ codes, and NVCC 11.7 compiler for CUDA codes.
The C++ codes are C++17 compliant.

## Restrictions
Under CUDA Compute Capability prior to 6 (Pascal), objects of type "cuda::atomic" may not be used.

## Generate expected results from given data
Run Jupyter Notebooks on Google Colab to generate expected results. The default experiment data are in the `data/` directory. You can also run the experiments again to gather your data and change the file path in Jupyter Notebooks. 

## Run partial experiments
### Step 1
Run C++ codes through 1 input
> sh run_one_cpp.sh

### Step 2
Read the output and convert to csv files
> sh generate_cpp_csv.sh

### Step 3
Run the /analysis/analysis_cpp.ipynb to create figures for C++ results

## How to run the experiments

### Step 1
Download the larger inputs and run all the codes through 5 inputs.
> sh run_all.sh

If the whole experiments take too long to complete, you can run partial results using `run_cpp.sh`.

### Step 2
Read the throughputs and convert them to csv files for each programming model.
> sh generate_*_csv.sh

For example, `generate_cpp_csv.sh` generates the csv files for C++ programs' throughputs.

### Step 3
Run the notebook to calculate metrics and generate figures for performance analysis.
