import tensorflow as tf
import os
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
from ruamel.yaml import YAML
import json

def load_params():
    "Updates FULL_PARAMS with the values in params.yaml and returns all as a dictionary"
    yaml = YAML(typ="safe")
    with open("params.yaml") as f:
        params = yaml.load(f)
    return params

params = load_params()

img_height = 224 #@param {type:'integer'}
img_width = 224 #@param {type:'integer'}

test_list_ds_1 = tf.data.TextLineDataset(["dataset/test.txt"])

test_ds = test_list_ds_1

num_test = len(list(test_ds))

AUTOTUNE = tf.data.AUTOTUNE

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  
  #age = int(tf.strings.split(parts[-1], "_")[0])
  age = int(parts[-2])

  #tf.print(parts)
  # Integer encode the label
  return age

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img) 
  return img, label


def configure_for_performance(ds):
  #ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(params["test"]["batch_size"])
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

test_ds = configure_for_performance(test_ds)

def build_model():

    inputs = layers.Input(shape=(img_height, img_width, 3))

    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    print("Number of layers in the base model: ", len(model.layers))

    model.trainable = False

    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization(name="our_bn")(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    outputs = layers.Dense(1, activation="linear", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer, loss="mae", metrics=[]
    )
    return model

model = build_model()

model.load_weights("models/best_val_loss.h5")

optimizer = tf.keras.optimizers.Adam(learning_rate=params["train"]["lr"])
model.compile(
    optimizer=optimizer, loss="mae", metrics=["mse", "mae"]
)

res =model.evaluate(
  test_ds,
  steps= num_test // params["test"]["batch_size"],)

print("testing done!")

print(res)

if not os.path.exists("runs"):
  os.makedirs("runs")

obj = {"mae" : res[0][0], "mse": res[0][1]}
with open("runs/test.json", "w") as f:
  json.dump(obj, f)