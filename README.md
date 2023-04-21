# Median housing value prediction

The housing data can be downloaded from https://github.com/ageron/handson-ml/tree/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script

## Setup the environment
- **with virtualenv (recommended)**
  - create environment using env.yml file: `conda env create -f env.yml`
  - activate the environment: `conda activate mle-dev`


## run the script

 - script run command: `python nonstandardcode.py`

## run the HOUSING_PRICE scripts

  - activate the environment: `conda activate mle-dev`
  - Move to HOUSING_PRICE/src folder :  `cd HOUSING_PRICE/src`
  - run script to create train-test data  : `python ingest_data.py`
  - run script to train data  : `python train.py`
  - run script to test data : `python score.py`

  Instead of running the scripts separately, directly run the command:

  - run final_script : `python final_script_ml_flow.py`


 ## run setup file to create module for tests
  - Move to HOUSING_PRICE folder :  `cd HOUSING_PRICE`
  - run setup.py file to build src module : `python setup.py install`

 ## run tests

  - Move to HOUSING_PRICE/tests folder :  `cd HOUSING_PRICE/tests/functional_tests`
  - run functional tests scripts  : `python <functional_test_file_name>.py`

  - Move to HOUSING_PRICE/tests folder :  `cd HOUSING_PRICE/tests/unit_tests`
  - run unit tests scripts  : `python <unit_test_file_name>.py`

  - Move to HOUSING_PRICE/tests folder :  `cd HOUSING_PRICE/tests/initial_tests`
  - run basic tests scripts  : `python <initial_test_file_name>.py`
