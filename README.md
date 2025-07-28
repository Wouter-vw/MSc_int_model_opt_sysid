# Self-Identifying Internal Model-Based Online Optimization

This repository houses all code, results and figures used throughout the master thesis project titled "Self-Identifying Internal Model-Based Online Optimization". The contents of the included files and folders is detailed below. 

## Python Files
In the main folder of the repository there are a number of python files, each with seperate functions. These files require a small number of dependencies which, for completeness are given in `online_opt.yaml'. To create this virtual environment do the following: 
```
conda env create -f online_opt.yaml
```
Activate the environment with:
```
conda activate online_opt
```
### Execution
To test the algorithm on new data, `Execution.py' can be run. In this file a number of variables can be specified, and b_type and A_type must be specified. If sava_data is set to true, the existing data will be overwritten. SIMBO's input variables are determined by the line:
```
online_gradient_control_sys_id_ARM(f, [mu,L], e_threshold = , e_threshold2 = , delta = , f_factor1 =, f_factor2 = , win_size1=, win_size2=, step=, x_0=)
```
### SYSID
"SYSID.py" houses the code necessary to run the algorithm, including the recursive least squares component and the internal model-based optimizer. Additionally, "tools.py" house important functions which are used including cost definition and controller recomputation.

### Remaining Python Files
The remaining python files are used to generate the tracking error plots for all test cases.

## Data
The folder "data" houses the saved results and plots for all cases in which the algorithm is tested. The plots include all algorithms used as a comparison and are not the versions used in the final thesis.

## Known $\boldsymbol b_k$
This folder houses its own data folder and python files. It is essentially the same as this main folder, however it assumes that $\boldsymbol b_k$ is known as a measurement. 

> Note: Running the code that is in this folder may not work due to altered path dependencies as a result of including it in its own seperate folder!

## Thesis Plots
This folder contains all the plots displayed in the thesis. These plots may show less comparison algorithms or simply have different titles than the ones in the folder data. It should be noted that the underlying data between the plots here and the ones in the folder "data" is the same.

Should any questions arise about the data, the algorithm or the thesis itself, feel free to get in touch. 
