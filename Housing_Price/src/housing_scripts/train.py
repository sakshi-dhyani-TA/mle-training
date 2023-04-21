# Importing  libraries
import argparse
import logging
import logging.config
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

# Setting up a config

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "DEBUG"},
}

# configuration of logger


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Parameters
    ----------
            logger:
                    Predefined logger object if present. If None a ew logger object will be created from root.
            cfg: dict()
                    Configuration of the logging to be implemented by default
            log_file: str
                    Path to the log file for logs to be stored
            console: bool
                    To include a console handler(logs printing in console)
            log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    Returns
    -------
    logging.Logger
    """
    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)

    return logger


def train_data(input_path, output_path):

    """This function uses training and test data for model creation

    Parameters
    ----------
    input_path : str
        Path to the input file

    output_path : str
        Path to the output file

    returns
    -------
    bool
        True if function completes successfully

    """

    # import training data
    success = False
    strat_train_set = pd.read_csv(input_path + "/train.csv")
    strat_test_set = pd.read_csv(input_path + "/test.csv")

    # dropping labels

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # handling data using imputer

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

    # Feature transformation

    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    # Linear Regression Model

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    # prediction for training data

    housing_predictions = lin_reg.predict(housing_prepared)

    # MSE value
    lin_mse = mean_squared_error(housing_labels, housing_predictions)

    # RMSE value
    lin_rmse = np.sqrt(lin_mse)
    print("rmse: ", lin_rmse)

    # MAE value
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    print("mae: ", lin_mae)

    # Decision Tree Regressor Model

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    # Prediction
    housing_predictions = tree_reg.predict(housing_prepared)

    # MSE value
    tree_mse = mean_squared_error(housing_labels, housing_predictions)

    # RMSE value
    tree_rmse = np.sqrt(tree_mse)
    print("rmse: ", tree_rmse)

    # Random Forest Regressor
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)

    # RandomizedSearchCV

    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_

    # mean scores

    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # GridSearchCV

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    # Random Forest Regressor

    forest_reg = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    grid_search.best_params_
    cvres = grid_search.cv_results_

    # Printing mean scores
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    # Final Model will be created using Grid Search CV

    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )

    # Feature transformation for test data

    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    # saving model file

    joblib.dump(final_model, output_path + "/housing_model.pkl")
    success = True
    return success
    print("Success! ")


if __name__ == "__main__":

    """___main___ function"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        help="Provide folder path",
        required=False,
        nargs="?",
        default=os.path.join(os.path.dirname(os.getcwd()), r"data\processed"),
    )
    parser.add_argument(
        "-o",
        "--out_path",
        help="Provide output folder path",
        required=False,
        nargs="?",
        default=os.path.join(os.path.dirname(os.getcwd()), r"artifacts"),
    )
    parser.add_argument("--log-level", help="specify the log level")
    parser.add_argument("--log-path", help="use a log file or not")
    parser.add_argument(
        "--no-console-log",
        help="toggle whether or not to write logs to the console",
        action="store_true",
    )
    console_log = True
    log_level = "DEBUG"
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.out_path
    if args.log_level:
        log_level = args.log_level
    if args.log_path:
        log_file = args.log_path
    else:
        log_file = "../logs/train_script_logs.log"
    if args.no_console_log:
        console_log = False

    # configuring and assigning in the logger can be done by the below function
    logger = configure_logger(
        log_file=log_file, console=console_log, log_level=log_level
    )
    logger.info(time.ctime(time.time()))
    logger.info("Function Started. Training Of the data started!")
    success = train_data(input_path, output_path)
    logger.info(time.ctime(time.time()))
    logger.info("Success! Model created and stored! Check artifacts!")
