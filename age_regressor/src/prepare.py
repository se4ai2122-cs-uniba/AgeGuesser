# iterate through all images and create train/test/validation splits with given ratio
# returns 3 txts with image paths

import glob
from ruamel.yaml import YAML
import random

def load_params():
    "Updates FULL_PARAMS with the values in params.yaml and returns all as a dictionary"
    yaml = YAML(typ="safe")
    with open("params.yaml") as f:
        params = yaml.load(f)
    return params

params = load_params()

all_imgs = []

folders = ["dataset/train/*/*.jpg", "dataset/validation/*/*.jpg", "dataset/test/*/*.jpg"]

for f in folders:
  all_imgs.extend(glob.glob(f))

for i, img in enumerate(all_imgs):
  all_imgs[i] = "/".join(img.split("/")[-4:])

print(all_imgs[0])

random.shuffle(all_imgs)

tot = len(all_imgs)
print(tot)
tot_train = int(tot*params["prepare"]["train"])

train_files = all_imgs[:tot_train]
test_files = all_imgs[tot_train:]

valid_split = int(len(train_files)*params["prepare"]["validation"])
valid_files = train_files[:valid_split]
train_files = train_files[valid_split:]

with open("dataset/train.txt", "w") as f:
  for l in train_files:
    f.write(l+"\n")

with open("dataset/test.txt", "w") as f:
  for l in test_files:
    f.write(l+"\n")

with open("dataset/validation.txt", "w") as f:
  for l in valid_files:
    f.write(l+"\n")