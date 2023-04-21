# importing libraries
import warnings
import mlflow
import mlflow.sklearn
import os
import tarfile
import urllib
import pandas as pd
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings('ignore')
# fetching data urls

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# function to fetch data

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#function to load data

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# loading data  using function

housing = load_housing_data()

#  small transformer class that adds the combined attributes

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



with mlflow.start_run(run_name = 'PARENT_RUN') as parent_run:
    with mlflow.start_run(run_name='data_prep_step', nested=True) as data_prep_run:
        mlflow.log_param("child", "yes")
        housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # creating copy of train set
    housing = strat_train_set.copy()

    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"]=housing["population"]/housing["households"]

    # separating target variable and predictors
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # to use imputer copying data without textual feature

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)

    # trained imputer used to transform the data by replacing null values

    X = imputer.transform(housing_num)

    # converting plain numpy array result into dataframe

    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                            index=housing_num.index)

    housing_cat = housing[["ocean_proximity"]]

    # using one hot encoder to ensure proper representation for categories
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

    with mlflow.start_run(run_name='data_transform_step', nested=True) as data_transform_run:
        attr_adder = CombinedAttributesAdder()
        housing_extra_attribs = attr_adder.transform(housing.values)

        # pipeline to transform data

        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])

        housing_num_tr = num_pipeline.fit_transform(housing_num)
        # Applying Column transformer

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]

        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])

        housing_prepared = full_pipeline.fit_transform(housing)

    with mlflow.start_run(run_name='data_training_step', nested=True) as data_training_run:
        forest_reg = RandomForestRegressor()
        param_grid = {
        'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8],
        'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]
        }
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                            scoring='neg_mean_squared_error',
                            return_train_score=True)
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        X_test_prepared = full_pipeline.transform(X_test)
        grid_search.fit(housing_prepared, housing_labels)

        best_rf = grid_search.best_estimator_
        for p in param_grid:
            print("Best '{}': {}".format(p, best_rf.get_params()[p]))

        predictions = best_rf.predict(X_test_prepared)
        # Log model
        mlflow.sklearn.log_model(best_rf, "grid-random-forest-model")
        # Log params
        model_params = best_rf.get_params()
        [mlflow.log_param(p, model_params[p]) for p in param_grid]
        (rmse, mae, r2) = eval_metrics(y_test, predictions)
        print("rmse:",rmse)
        print("mae:",mae)
        # Create and log MSE metrics using predictions of X_test and its actual value y_test
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae",mae)
        mlflow.sklearn.log_model(forest_reg, "model")