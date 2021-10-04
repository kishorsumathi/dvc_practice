import yaml
import os

def read_yaml(path_to_yaml:str):
    with open(path_to_yaml) as yaml_file:
        content= yaml.safe_load(yaml_file)
    
    return content
def create_directory(dirs: list):
    for dir_path in dirs:
          os.makedirs(dir_path,exist_ok=True)
          print(f"directory is created at {dir_path}")

def save_local_df(df,path,index_status=False):
    df.to_csv(path,index=index_status)
    print(f"data is saved at {path}")