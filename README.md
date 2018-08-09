# Canary Anomaly Detection

## Installation

You should just install the requirements:
```bash
pip install -e .
```

## Testing

You can run tests by typing `pytest` in the `canary` directory.

## Generator
The generator module is use to generate the test data, i.e. data with synthetic anomalies. 
The Data flow:
1. Download the data with `download_whole.py`, eg:
    ```bash
    python canary/generator/download/download_whole.py /some/directory --nightly_version_number 5
    ``` 
    The script makes catalogs in `/some/directory` (first argument)  with the data from 5 (second argument) 
    latest versions of nightly. The data is downloaded from Telemetry 
    [HTTP API](https://github.com/mozilla/python_mozaggregator#api)
2. Generate the data with anomalies with `generate_test_data.py`, eg:
    ```bash
    python canary/generator/generate_test_data.py example_data/*.json /some/directory --plots
    ```
    The first argument is the directory with downloaded data. In the example `example_data` from the module 
    is used. The second argument indicates the directory, where the generated data is saved. 
    The `--plots` flag specifies that the plots should be saved.
    
    What's actually happening to the data inside:
    * The data is preprocessed and split into train and test set. The `y` is generated on the assumption, 
    that there are no anomalies in the data set. The train set won't be changed.
    * The test set is changed by the pipelines, which are in `pipelines_*.py`. Each pipeline consists
    of some transformers, that are in `transformers` directory. Each pipeline (and each transformer) operates
    only on some kinds of the histograms. Some of the transformers add anomalies and change `y` and some only 
    add noise or trend.
    * The plots of changes are generated with usage of the `plot` function from `utils`.
    * Everything is saved in the directory provided by the user. In our example `/some/directory`.