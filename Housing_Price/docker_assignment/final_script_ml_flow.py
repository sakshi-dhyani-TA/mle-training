import argparse
import logging
import logging.config
import os
import time

import ingest_data
import mlflow
import score
import train

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

# logger configuration


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


def mlflow_func():
    logger.info(time.ctime(time.time()))
    logger.info("Data Preparation Started!! ")

    # or set the MLFLOW_TRACKING_URI in the env
    # remote_server_uri = "http://localhost:8080"  # set to your server URI
    # mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env
    experiment = mlflow.set_experiment("Housing Price Prediction")

    with mlflow.start_run(
        run_name="Data Preparation",
        nested=True,
        experiment_id=experiment.experiment_id,
        tags={"version": "v1"},
    ) as parent_run:
        mlflow.log_param("Parent", "yes")
        out = ingest_data.process_data(
            os.path.join(os.path.dirname(os.getcwd()), r"data\processed")
        )

        print("Data Preparation Completed!! ")
        mlflow.log_metric("metrics ", out)
        logger.info(time.ctime(time.time()))
        logger.info("Data Preparation Done!! ")

        logger.info(time.ctime(time.time()))
        logger.info("Data Training Started!! ")

        with mlflow.start_run(
            run_name="Data Training",
            nested=True,
            experiment_id=experiment.experiment_id,
        ) as child_run:
            mlflow.log_param("Child 1", "yes")
            print("Data Training Started !! ")
            out = train.train_data(
                input_path=os.path.join(
                    os.path.dirname(os.getcwd()), r"data\processed"
                ),
                output_path=os.path.join(os.path.dirname(os.getcwd()), r"artifacts"),
            )
            mlflow.log_metric("metrics ", out)
            print("Data Training Completed!! ")
            logger.info(time.ctime(time.time()))
            logger.info("Data Training Done!! ")

        with mlflow.start_run(
            run_name="Model_Scoring",
            experiment_id=experiment.experiment_id,
            nested=True,
        ) as child_run:
            mlflow.log_param("Child 2", "yes")
            print("Model Performance !! ")
            rmse, mae, r2 = score.metrics_evaluation(
                input_path=os.path.join(
                    os.path.dirname(os.getcwd()), r"data\processed"
                ),
                output_path=os.path.join(os.path.dirname(os.getcwd()), r"Score"),
                model_path=os.path.join(os.path.dirname(os.getcwd()), r"artifacts"),
            )
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("R2 Score", r2)
            print("Scores Evaluated. Check Score file !! ")
            logger.info(time.ctime(time.time()))
            logger.info("Metrics Evaluated!! ")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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

    if args.log_level:
        log_level = args.log_level
    if args.log_path:
        log_file = args.log_path
    else:
        log_file = "final_script_logs.log"
    if args.no_console_log:
        console_log = False

    logger = configure_logger(
        log_file=log_file, console=console_log, log_level=log_level
    )
    logger.info(time.ctime(time.time()))
    logger.info("Running Final Script!! ")
    mlflow_func()
    logger.info(time.ctime(time.time()))
    logger.info("Success! Metrics Evaluated! ")
