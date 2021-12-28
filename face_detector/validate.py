import os, sys

p = os.path.abspath('..')
sys.path.insert(1, p)
from data_expectations import create_dataset_file, dataset_validation
import glob

with open("dataset/yolo/state.txt", "r") as f:
  uid = f.read()

train_imgs = glob.glob("dataset/yolo/train/images/*")
valid_imgs = glob.glob("dataset/yolo/valid/images/*")
test_imgs = glob.glob("dataset/yolo/test/images/*")     

splits = [{
  "meta": "dataset/images_face_detection_train.csv",
  "data": "dataset/yolo/train/images/*"
  }, 
  {
  "meta": "dataset/images_face_detection_valid.csv",
  "data": "dataset/yolo/valid/images/*"
  }, 
  {
  "meta": "dataset/images_face_detection_test.csv",
  "data": "dataset/yolo/test/images/*"
  }, ]

partial_success = True

for split in splits:
  imgs = glob.glob(split["data"])
  create_dataset_file.create(split["meta"], imgs)
  results = dataset_validation.test_ge(split["meta"])
  
  print(results)
  
  
  for result in results:
    print(result["success"])
    partial_success = partial_success and result["success"]

  if not partial_success:
    break

with open("dataset/data_valid_result.txt", "w") as f:
  f.write(uid.strip() + "-" + str(partial_success) )

assert partial_success

"""
images_face_detection_train = glob.glob("dataset/yolo/train/images/*")
images_face_detection_valid = glob.glob("dataset/yolo/valid/images/*")
images_face_detection_test = glob.glob("dataset/yolo/test/images/*")     
                
create("images_face_detection_train.csv",images_face_detection_train)                
create("images_face_detection_valid.csv",images_face_detection_valid)                
create("images_face_detection_test.csv",images_face_detection_test) 

test_ge("images_face_detection_train.csv")
test_ge("images_face_detection_valid.csv")
test_ge("images_face_detection_test.csv")
"""