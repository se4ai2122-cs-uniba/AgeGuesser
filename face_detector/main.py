import torch

from IPython.display import Image, clear_output  # to display images
#from utils.google_utils import gdrive_downl#ad  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


dataset_base = "/content/drive/MyDrive/AI Playground/face_detector/dataset/faces/"


dataset_yolo = dataset_base + "yolo/"

data_yaml = dataset_yolo + "data.yaml"

#pretrained = "/content/drive/MyDrive/AI Playground/face_detector/yolov5s.pt"

#trained_custom = "/content/drive/MyDrive/AI Playground/face_detector/dataset/faces/best_l.pt"

test_path = dataset_base + "yolo/test/images"

# define number of classes based on YAML
import yaml
with open("dataset/yolo/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
        
        
        