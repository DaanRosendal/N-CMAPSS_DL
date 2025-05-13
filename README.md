# N-CMAPSS_DL
DL evaluation on N-CMAPSS
Turbo fan engine           |  CMAPSS [[1]](#1)
:----------------------------:|:----------------------:
![](turbo_engine.jpg)  |  ![](cmapss.png)

## Prerequisites

### Environment Setup
1. Install uv (Python package manager)

For Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
# or
.\.venv\Scripts\activate   # On Windows
```

3. Install dependencies:
```bash
uv sync
```

## Sample creator
Following the below instruction, you can create training/test sample arrays for machine learning model (especially for DL architectures that allow time-windowed data as input) from NASA's N-CMAPSS datafile. <br/>
Please download Turbofan Engine Degradation Simulation Data Set-2, so called N-CMAPSS dataset [[2]](#2), from [NASA's prognostic data repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). In case the link does not work, please temporarily use this [shared drive](https://drive.google.com/drive/folders/1HtnDBGhMoAe53hl3t1XeCO9Z8IKd3-Q-)
<br/>
In the downloaded dataset, dataset DS01 has been used for the application of model-based diagnostics and dataset DS02 has been used for data-driven prognostics.   Therefore, we need only dataset DS02. <br/>
Please locate "N-CMAPSS_DS02-006.h5"file to /N-CMAPSS folder. <br/>
Then, you can get npz files for each of 9 engines by running the python codes below.
```bash
uv run sample_creator_unit_auto.py -w 50 -s 1 --test 0 --sampling 10
```
After that, you should run
```bash
uv run sample_creator_unit_auto.py -w 50 -s 1 --test 1 --sampling 10
```
&ndash;  w : window length <br/>
&ndash;  s : stride of window <br/>
&ndash;  test : select train or test, if it is zero, then the code extracts samples from the engines used for training. Otherwise, it creates samples from test engines<br/>
&ndash;  sampling : subsampling the data before creating the output array so that we can set assume different sampling rate to mitigate memory issues.


Please note that we used N = 6 units (u = 2, 5, 10, 16, 18 & 20) for training and M = 3  units (u = 11, 14 & 15) for test, same as for the setting used in [[3]](#3). <br/>

The size of the dataset is significantly large and it can cause memory issues by excessive memory use. Considering memory limitation that may occur when you load and create the samples, we set the data type as 'np.float32' to reduce the size of the data while the data type of the original data is 'np.float64'. Based on our experiments, this does not much affect to the performance when you use the data to train a DL network. If you want to change the type, please check 'data_preparation_unit.py' file in /utils folder.  <br/>

In addition, we offer the data subsampling to handle 'out-of-memory' issues from the given dataset that use the sampling rate of 1Hz. When you set this subsampling input as 10, then it indicates you only take only 1 sample for every 10, the sampling rate is then 0.1Hz.

Finally, you can have 9 npz file in /N-CMAPSS/Samples_whole folder. <br/>

Each compressed file contains two arrays with different labels: 'sample' and 'label'. In the case of the test units, 'label' indicates the ground truth RUL of the test units for evaluation.

For instance, one of the created file, Unit2_win50_str1_smp10.npz, its filename indicates that the file consists of a collection of the sliced time series by time window size 50 from the trajectory of engine (unit) 2 with the sampling rate of 0.1Hz. <br/>

## Running Scripts

All scripts can be run using `uv run` to ensure they use the correct Python environment:

```bash
uv run script_name.py [arguments]
```

### Script Overview

1. **train_rul_model_lightweight.py**: Trains a simple MLP model on the N-CMAPSS dataset for RUL prediction.
   ```bash
   uv run train_rul_model_lightweight.py
   ```
   This script loads sample data from training and test units, normalizes the data, trains a simple neural network model, and evaluates its performance on test data. The model and evaluation metrics are saved to disk.

2. **data_inspector.py**: Inspects and analyzes NPZ files from the N-CMAPSS dataset.
   ```bash
   uv run data_inspector.py path/to/file.npz [--filter-range start:end] [--filter-rul threshold] [--output-file output.npz]
   ```
   This tool allows you to examine the structure and contents of the dataset files, visualize RUL distributions, and create filtered subsets based on index ranges or RUL thresholds.

3. **data_filter_by_cycle.py**: Filters data based on engine cycle information.
   ```bash
   uv run data_filter_by_cycle.py --npz-file path/to/file.npz --unit unit_number [--show-cycles] [--max-cycle max_cycle] [--output-file output.npz]
   ```
   This script helps filter data by engine cycle number, which is useful for creating datasets with specific operating conditions or degradation stages.

4. **prepare_federated_data.py**: Prepares data for federated learning experiments.
   ```bash
   uv run prepare_federated_data.py [--data-dir dir] [--output-dir dir] [--n-clients N] [--split-method method] [--train-units units] [--rul-threshold threshold] [--max-cycles unit:cycle,...]
   ```
   This script divides the dataset among multiple simulated clients using different distribution strategies (by unit, by RUL range, or randomly) for federated learning research.

## References
<a id="1">[1]</a>
Frederick, Dean & DeCastro, Jonathan & Litt, Jonathan. (2007). User's Guide for the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS). NASA Technical Manuscript. 2007–215026.

<a id="2">[2]</a>
Chao, Manuel Arias, Chetan Kulkarni, Kai Goebel, and Olga Fink. "Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics." Data. 2021; 6(1):5. https://doi.org/10.3390/data6010005

<a id="3">[3]</a>
Chao, Manuel Arias, Chetan Kulkarni, Kai Goebel, and Olga Fink. "Fusing physics-based and deep learning models for prognostics." Reliability Engineering & System Safety 217 (2022): 107961.

<a id="3">[4]</a>
Mo, Hyunho, and Giovanni Iacca. "Multi-objective optimization of extreme learning machine for remaining useful life prediction." In Applications of Evolutionary Computation: 25th European Conference, EvoApplications 2022, Held as Part of EvoStar 2022, Madrid, Spain, April 20–22, 2022, Proceedings, pp. 191-206. Cham: Springer International Publishing, 2022.

Bibtex entry ready to be cited
```
@inproceedings{mo2022multi,
  title={Multi-objective optimization of extreme learning machine for remaining useful life prediction},
  author={Mo, Hyunho and Iacca, Giovanni},
  booktitle={Applications of Evolutionary Computation: 25th European Conference, EvoApplications 2022, Held as Part of EvoStar 2022, Madrid, Spain, April 20--22, 2022, Proceedings},
  pages={191--206},
  year={2022},
  organization={Springer}
}
```
