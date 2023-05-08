#Deriving the latest base image
FROM python:latest


# Any working directory can be chosen as per choice like '/' or '/home' etc

WORKDIR /docker_assignment

#to COPY the remote file at working directory in container

COPY final_script_ml_flow.py ./
COPY ingest_data.py ./
COPY housing.csv ./
COPY score.py ./
COPY train.py ./
COPY train.csv ./
COPY test.csv ./

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install the required packages

RUN pip install numpy \
    pandas \
    scikit-learn \
    mlflow		     



#CMD instruction should be used to run the software
#contained by your image, along with any arguments.

CMD [ "python", "./final_script_ml_flow.py"]
