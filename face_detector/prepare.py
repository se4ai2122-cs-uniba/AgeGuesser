import shutil
import os
import random
import math
import yaml

def split_dataset(dataset_path,train_percentage, validation_percentage,classes=[],mode="move",output_path=""):
  def shift(source,destination,mode):
    if mode == "copy":
      shutil.copyfile(source,destination)
    else:
      shutil.move(source,destination) 
       
  def diff_lists(l1,l2):
    from collections import Counter
    return list((Counter(l1) - Counter(l2)).elements())
  
  if mode == "copy" and output_path == "":
    raise Exception("Cannot copy files on the same directory")  
      
  if validation_percentage > train_percentage:
    raise Exception("validation_percentage must be lower than train_percentage")  
  
  other_files = None
  
  if classes == []:
    classes = os.listdir(dataset_path)
  else: 
    other_files = diff_lists(os.listdir(dataset_path),classes)      
  
  test_percentage = 1 - train_percentage
      
  if not dataset_path.endswith("/"):
    dataset_path = dataset_path + "/"

  if output_path == "":
    output_path = dataset_path
    
  train_dir = output_path + "/train"
  test_dir = output_path + "/test"
  validation_dir = output_path + "/valid"
  seed = 42
    
  for _class in classes:
    os.makedirs(train_dir + "/" + _class,exist_ok=True)
    os.makedirs(test_dir + "/" + _class,exist_ok=True)
    os.makedirs(validation_dir + "/" + _class,exist_ok=True)    
        
    data = sorted(os.listdir(dataset_path + _class + "/"))
    random.Random(seed).shuffle(data)
    
    data_length = len(data)
    
    test_size = math.floor(data_length * test_percentage)
    validation_size = math.floor(data_length * validation_percentage)

    for i,single_data in enumerate(data):
          
      single_data_path = dataset_path + _class + "/" + single_data    
        
      if i < test_size:
        shift(single_data_path, test_dir +  "/" + _class + "/" + single_data, mode)
                    
      elif test_size < i <= test_size + validation_size:
        shift(single_data_path, validation_dir + "/" + _class + "/" + single_data, mode)
        
      else:    
        shift(single_data_path, train_dir + "/" + _class + "/" + single_data, mode)
    if mode == "move":
      shutil.rmtree(dataset_path + _class)      
  
  if other_files is not None:
    for file in other_files:
      shift(dataset_path + file, output_path + "/" + file, mode) 

os.system("unzip -n dataset.zip")

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
    
    train_percentage = params['preparation']['train_percentage']
    validation_percentage = params['preparation']['validation_percentage']
       
    split_dataset(dataset_path="dataset/yolo",train_percentage=train_percentage,validation_percentage=validation_percentage,classes=["images","labels"])
