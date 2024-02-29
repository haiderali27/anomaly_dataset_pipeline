# anomaly_dataset_pipeline
This Repository contains the pipeline script for Cars Anomaly Detection Image Dataset. The dataset is heterogeneous in nature. Main purpose is for dataset is to detect anomalies in heterogeneous dataset.  Since, Car is more heterogeneous that industrial objects. Dataset structure is analogous to Mvtec Dataset class.

## About
- [Pipeline Details](#Pipeline-Details)
- [Run](#Run-The-Code)

## Pipeline Details
`Pipeline is divided in to two parts. First part is data gathering. Second part is segmenting the images, so it only focus on the target class object`
`Sam Implementation has many parameters, all of them are for how data should be processed. Most of names are self explinatory`
`There can be many issue that can happen when you run this pipeline, make sure your system has cuda12.1+ and all drivers install before running requirements file.`

## Run The Code
1. `git clone https://github.com/haiderali27/anomaly_dataset_pipeline.git` or `git clone git@github.com:haiderali27/anomaly_dataset_pipeline.git`
2. Run the requirement File `pip install -r requirements.txt`
3. Run `python pipeline_script.py` Make sure you put your kaggle authentication correctly and gdown will not download file for next hour since dataset from gdown is 5.6 GB. 
4. You can run both of scripts explicitly `python prepare_data.py` and `python segment_data.py`

