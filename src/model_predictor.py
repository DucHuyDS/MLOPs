import argparse
import logging
import os
import random
import time

import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object
from pydantic import BaseModel
from multiprocessing import cpu_count
from problem_config import ProblemConst, create_prob_config
from raw_data_processor import RawDataProcessor
from utils import AppConfig, AppPath

PREDICTOR_API_PORT = 8000


class Data(BaseModel):
    id: str
    rows: list
    columns: list


class ModelPredictor:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info(f"model-config: {self.config}")

        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)

        # load model
        model_uri = os.path.join(
            "models:/", self.config["model_name"], str(self.config["model_version"])
        )
        self.model = mlflow.pyfunc.load_model(model_uri)

    def detect_drift(self, feature_df) -> int:
        #1: drift, 0: no drift
        if feature_df['feature37'].isna().sum() > len(feature_df)*0.3 and feature_df['feature4'].isna().sum() > len(feature_df)*0.3:
            return 1
        return 0

    def predict(self, data: Data):
        start_time = time.time()

        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        # save request data for improving models
        ModelPredictor.save_request_data(
            feature_df, self.prob_config.captured_data_dir, data.id
        )

        prediction = self.model.predict(feature_df)
        is_drifted = self.detect_drift(feature_df)

        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction takes {run_time} ms")
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
        feature_df.to_parquet(output_file_path, index=False)
        return output_file_path


class PredictorApi:
    def __init__(self, predictor: ModelPredictor):
        self.predictor = predictor
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/phase-1/prob-1/predict")
        async def predict(data: Data, request: Request):
            response = self.predictor.predict(data)
            return response


    def run(self, port):
        number_of_workers = cpu_count() * 2 + 1
        uvicorn.run(
            "model_predictor:api.app",
            host="0.0.0.0",
            port=port,
            workers=number_of_workers,
        )



default_config_path = (
    AppPath.MODEL_CONFIG_DIR
    / ProblemConst.PHASE1
    / ProblemConst.PROB1
    / "model-1.yaml"
).as_posix()

parser = argparse.ArgumentParser()
parser.add_argument("--config-path", type=str, default=default_config_path)
parser.add_argument("--port", type=int, default=PREDICTOR_API_PORT)
args = parser.parse_args()

predictor = ModelPredictor(config_file_path=args.config_path)
api = PredictorApi(predictor)
if __name__ == "__main__":
    api.run(port=args.port)
