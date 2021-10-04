from scipy.sparse.construct import random
from src.utils.all_utils import create_directory, read_yaml, save_local_df,save_model
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import os

def training(config_path,params_path):
      config =read_yaml(config_path)
      params= read_yaml(params_path)
      artifacts_dir=config["artifacts"]["artifacts_dir"]
      split_data_dir=config["artifacts"]["split_data_dir"]
      train_data_filename=config["artifacts"]["train"]
      model_dir=config["artifacts"]["model_dir"]
      model_path=config["artifacts"]["saved_model"]
      train_data_filename_test=config["artifacts"]["test"]
      alpha_1=params["model_params"]["Elastic_net"]["alpha"]
      l1_ratio_1=params["model_params"]["Elastic_net"]["l1_ratio"]
      random_state_1=params["model_params"]["Elastic_net"]["random_state_elastic"]
      train_data_path= os.path.join(artifacts_dir,split_data_dir,train_data_filename)
      test_data_path= os.path.join(artifacts_dir,split_data_dir,train_data_filename_test)
      train_data= pd.read_csv(train_data_path)
      test= pd.read_csv(test_data_path)
      train_y=train_data["quality"]
      train_x=train_data.drop("quality",axis=1)
      model= ElasticNet(alpha=alpha_1,l1_ratio=l1_ratio_1,random_state=random_state_1)
      model.fit(train_x,train_y)
      model_dir1=os.path.join(artifacts_dir,model_dir)
      create_directory(dirs=[model_dir1])
      save_model1= os.path.join(artifacts_dir,model_dir,model_path)
      save_model(model,save_model1)



if __name__ == "__main__":
      args=argparse.ArgumentParser()
      args.add_argument("--config","-c",default="config/config.yaml")
      args.add_argument("--params","-p",default="params.yaml")
      
      parsed_args= args.parse_args()
      training(config_path=parsed_args.config,params_path=parsed_args.params)
      