import tensorflow as tf
import os
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, callbacks
from ruamel.yaml import YAML

def load_params():
    "Updates FULL_PARAMS with the values in params.yaml and returns all as a dictionary"
    yaml = YAML(typ="safe")
    with open("params.yaml") as f:
        params = yaml.load(f)
    return params

params = load_params()

img_height = 224 #@param {type:'integer'}
img_width = 224 #@param {type:'integer'}

train_list_ds_1 = tf.data.TextLineDataset(["dataset/train.txt"])
valid_list_ds_1 = tf.data.TextLineDataset(["dataset/validation.txt"])

train_ds = train_list_ds_1
valid_ds = valid_list_ds_1

num_train = len(list(train_ds))
num_valid = len(list(valid_ds))

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
  ds = ds.batch(params["train"]["batch_size"])
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)

valid_ds = valid_ds.map(process_path, num_parallel_calls=AUTOTUNE)

train_ds = configure_for_performance(train_ds)
valid_ds = configure_for_performance(valid_ds)

weights_path = "runs/"

  
if not os.path.exists(weights_path):
  os.makedirs(weights_path)

if not os.path.exists("models/"):
  os.makedirs("models/")

class CustomCallback(tf.keras.callbacks.Callback):
    ep_ = 1
    def on_epoch_begin(self, epoch, logs=None):
        print("epoch started")

    def on_epoch_end(self, epoch, logs=None):
        print("epoch ended")
        self.ep_ += 1

    def on_test_end(self, logs=None):
      # update mlflow metrics
      pass

    def on_train_batch_end(self, batch, logs=None):
      # update mlflow metrics
      pass

def scheduler(epoch, lr):
  lr_ = lr
  if epoch > 1:
    lr_ = lr_ * tf.math.exp(-0.15)

  return lr_

cs = [
      callbacks.ModelCheckpoint(filepath="models/best_val_loss.h5" ,
                                      monitor='val_loss',
                                      mode='min', save_weights_only=True, save_best_only=True),
      callbacks.ModelCheckpoint(filepath="models/best_loss.h5" ,
                                      monitor='loss',
                                      mode='min', save_weights_only=True, save_best_only=True),
      callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),
      callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_delta=1e-4, min_lr=0.0001),
      CustomCallback(),
      callbacks.LearningRateScheduler(scheduler, verbose=1)

  ]

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

def get_bin(age):
    if age <= 10:
      return 0
    elif age > 10 and age <= 20:
      return 1
    elif age > 20 and age <= 30:
      return 2
    elif age > 30 and age <= 40:
      return 3
    elif age > 40 and age <= 50:
      return 4
    elif age > 50 and age <= 60:
      return 5
    elif age > 60 and age <= 70:
      return 6
    elif age > 70 :
      return 7


optimizer = tf.keras.optimizers.Adam(learning_rate=params["train"]["lr"])
model.compile(
    optimizer=optimizer, loss="mae", metrics=["mse", "mae"]
)

model.fit(
  train_ds,
  steps_per_epoch= num_train // params["train"]["batch_size"],
  callbacks=cs,
  validation_data=valid_ds,
  validation_steps= num_valid // params["train"]["batch_size"],
  epochs=params["train"]["epochs"]
)

print("training done!")
