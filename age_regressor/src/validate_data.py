import os, sys

p = os.path.abspath('..')
sys.path.insert(1, p)
from data_expectations import create_dataset_file, dataset_validation
import glob

with open("dataset/state.txt", "r") as f:
  uid = f.read()

train_imgs = glob.glob("dataset/train_aug/*/*")
valid_imgs = glob.glob("dataset/validation/*/*")
test_imgs = glob.glob("dataset/test/*/*")     

splits = [{
  "meta": "dataset/meta_train.csv",
  "data": "dataset/train/*/*"
  }, 
  {
  "meta": "dataset/meta_validation.csv",
  "data": "dataset/validation/*/*"
  }, 
  {
  "meta": "dataset/meta_test.csv",
  "data": "dataset/test/*/*"
  }, ]

partial_success = True

for split in splits:
  imgs = glob.glob(split["data"])
  create_dataset_file.create(split["meta"], imgs)
  results = dataset_validation.test_ge(split["meta"])
  
  for result in results:
    partial_success = partial_success and result["success"]

  if not partial_success:
    break

with open("dataset/data_valid_result.txt", "w") as f:
  f.write(uid.strip() + "-" + str(partial_success) )

assert partial_success