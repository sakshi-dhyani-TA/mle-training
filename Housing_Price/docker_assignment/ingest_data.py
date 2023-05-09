# Importing required libraries
import argparse
import logging
import logging.config
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Setting up a config

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - \
            %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "DEBUG"},
}


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`,
    `log_level` will overwrite the ones in cfg.

    Parameters
    ----------
            logger:
                    Predefined logger object if present. If None a new
                    logger object will be created from root.
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


path = os.path.dirname(os.getcwd())


def process_data(file_path, inp_path=path):
    """This function creates training and testing data from the raw data

    Parameters
    ----------
    file_path : str
        Path to the output file

    returns
    -------
        bool
            True if function completes successfully
    """
    global strat_test_set
    global strat_train_set
    success = False

    #csv_path = os.path.join(inp_path, r"data\raw\housing.csv")

    #print("i_path :",inp_path)
    housing = pd.read_csv(r"housing.csv")

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    def income_cat_proportions(data):
        """This function is for generating income category proportions

        Parameters
        ----------
        data : DataFrame
            dataframe

        Returns
        -------
        Series

        """
        return data["income_cat"].value_counts() / len(data)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    strat_train_set.to_csv("/train.csv")
    strat_test_set.to_csv( "/test.csv")

    print("SUCCESS! Files Created!")
    success = True
    return success


if __name__ == "__main__":
    """___main___ function"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--out-path",
        help="Give folder for saving processed data! ",
        required=False,
        nargs="?",
        default=os.path.join(os.path.dirname(os.getcwd()), r"data\processed"),
    )
    parser.add_argument("--log-level", help="specify the log level")
    parser.add_argument("--log-path", help="use a log file or not")
    parser.add_argument(
        "--no-console-log",
        help="whether or not to write logs to the console",
        action="store_true",
    )
    console_log = True
    log_level = "DEBUG"
    args = parser.parse_args()
    file_path = args.out_path
    if args.log_level:
        log_level = args.log_level
    if args.log_path:
        log_file = args.log_path
    else:
        log_file = "ingest_data_logs.log"
    if args.no_console_log:
        console_log = False

    # logger configuration

    logger = configure_logger(
        log_file=log_file, console=console_log, log_level=log_level
    )
    logger.info(time.ctime(time.time()))
    logger.info("Function Started. Preparing Data Files!")
    success = process_data(file_path)
    logger.info(time.ctime(time.time()))
    logger.info(
        "Success! Files Created! Check processed folder for training and test datasets"
    )
