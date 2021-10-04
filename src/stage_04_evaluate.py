from scipy.sparse.construct import random
from src.utils.all_utils import create_directory, read_yaml, save_local_df,save_model,load_model, save_reports
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
import os

def evaluate_metrics(actual_value,predicted_values):
      rmse=np.sqrt(mean_squared_error(actual_value,predicted_values))
      mae=mean_absolute_error(actual_value,predicted_values)
      r2=r2_score(actual_value,predicted_values)
      return rmse,mae,r2
def evaluate(config_path):
      config =read_yaml(config_path)
      artifacts_dir=config["artifacts"]["artifacts_dir"]
      split_data_dir=config["artifacts"]["split_data_dir"]
      model_dir=config["artifacts"]["model_dir"]
      model_path=config["artifacts"]["saved_model"]
      report_dir=config["artifacts"]["report_dir"]
      report_path=config["artifacts"]["report"]
      train_data_filename_test=config["artifacts"]["test"]
      test_data_path= os.path.join(artifacts_dir,split_data_dir,train_data_filename_test)
      test= pd.read_csv(test_data_path)
      save_model1= os.path.join(artifacts_dir,model_dir,model_path)
      final_model=load_model(save_model1)
      test_x=test.drop(["quality"],axis=1)
      test_y=test["quality"]
      y_pred=final_model.predict(test_x)
      rmse,mae,r2 = evaluate_metrics(test_y,y_pred)
      report_dir1=os.path.join(artifacts_dir,report_dir)
      create_directory(dirs=[report_dir1])
      scores={
            "root_mean_squard_error": rmse,

            "mean_absolute_error": mae,

            "r_squared": r2,
      }
      final_path=os.path.join(artifacts_dir,report_dir,report_path)
      save_reports(scores,final_path)
      






if __name__ == "__main__":
      args=argparse.ArgumentParser()
      args.add_argument("--config","-c",default="config/config.yaml")
      
      parsed_args= args.parse_args()
      evaluate(config_path=parsed_args.config)
      