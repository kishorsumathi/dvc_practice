from scipy.sparse.construct import random
from src.utils.all_utils import create_directory, read_yaml, save_local_df
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(config_path,params_path):
      config =read_yaml(config_path)
      params= read_yaml(params_path)
      random_ste=params["base"]["random_state"]
      test_ratio=params["base"]["test_size"]
      remote_data_path=config["data_source"]
      artifacts_dir=config["artifacts"]["artifacts_dir"]
      raw_local_dir=config["artifacts"]["raw_local_dir"]
      raw_local_file=config["artifacts"]["raw_local_file"]
      split_data_dir=config["artifacts"]["split_data_dir"]
      train_data_filename=config["artifacts"]["train"]
      train_data_filename_test=config["artifacts"]["test"]
      create_directory(dirs=[os.path.join(artifacts_dir,split_data_dir)])

      raw_local_dir_path= os.path.join(artifacts_dir,raw_local_dir)
      data_path=os.path.join(raw_local_dir_path,raw_local_file)
      df = pd.read_csv(data_path)
      train,test=train_test_split(df,test_size=test_ratio,random_state=random_ste)
      train_data_path= os.path.join(artifacts_dir,split_data_dir,train_data_filename)
      test_data_path= os.path.join(artifacts_dir,split_data_dir,train_data_filename_test)
      save_local_df(train,train_data_path)
      save_local_df(test,test_data_path)

if __name__ == "__main__":
      args=argparse.ArgumentParser()
      args.add_argument("--config","-c",default="config/config.yaml")
      args.add_argument("--params","-p",default="params.yaml")
      
      parsed_args= args.parse_args()
      split_data(config_path=parsed_args.config,params_path=parsed_args.params)
      