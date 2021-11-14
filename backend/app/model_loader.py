import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import cv2
import base64
import io
import torch
import torchvision
from torchvision import transforms
from app.schemas import Face, FaceWithAge
import timm
from torch import nn


img_height = 224
img_width = 224

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    transforms.transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    torchvision.transforms.Resize((224, 224)),
])

def readb64_cv(base64_):
    msg = base64.decodebytes(bytes(base64_.split(",")[1], "utf-8"))
    buf = io.BytesIO(msg)
    pil_image = Image.open(buf).convert('RGB') 
    # pil_image.save('my-image.jpeg')
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    img = open_cv_image[:, :, ::-1].copy() 

    return img

def read_img(img_file):
    buf = io.BytesIO(img_file)
    pil_image = Image.open(buf).convert('RGB')
    # pil_image.save('my-image.jpeg')
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    img = open_cv_image[:, :, ::-1].copy() 

    return img


class EstimationModel(object):

  def __init__(self, weights_path:str,):
    self.weights_path = weights_path 
    self.type = "tf" if weights_path.endswith(".h5") else "torch"
    self.model = self.load_estimation_model()

  def load_estimation_model(self, ):

    if self.type == "tf":
      return self.load_effnet_model_tf()
    else:
      return self.load_effnet_model_torch()

  def load_effnet_model_torch(self,):
    class AgeNetwork(nn.Module):
      def __init__(self, backbone):
          super(AgeNetwork, self).__init__()
          
          self.backbone = backbone

          self.model1 = nn.Sequential(
              
              nn.Flatten(),
              nn.Dropout(0.2),
              nn.LazyLinear(1)
    
              
          )
          self.model2 = nn.Sequential(
              
              nn.Flatten(),
              nn.LazyLinear(6)
          )
            
      def forward(self, x, x_selfsup = None):
          y_ = self.backbone(x)
          y_sup = self.model1(y_)

          if x_selfsup is not None:
            y_self_ = self.backbone(x_selfsup)
            y_self = self.model2(y_self_)
            return y_sup, y_self
          return y_sup
    resnet = timm.create_model('tf_efficientnetv2_b0', pretrained=False)

    backbone = nn.Sequential(
        *list(resnet.children())[:-1]
    )

    for p in backbone.parameters():  # reset requires_grad
      p.requires_grad = False
    
    model = AgeNetwork(backbone)
    model.to("cpu")
    model.load_state_dict( torch.load(self.weights_path, map_location="cpu")["backbone_parameters"])
    return model

  def load_effnet_model_tf(self, ):

    inputs = layers.Input(shape=(224, 224, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights=None, )

    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization(name="our_bn")(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    outputs = layers.Dense(1, activation="linear", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="mae", metrics=[]
    )
    model.load_weights(self.weights_path)

    return model

  def predict(self, img):
    if self.type == "tf":
      return self.predict_tf(img)
    else:
      return self.predict_torch(img)

  def predict_tf(self, img ):
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    imgg = img.reshape(1, 224, 224, 3)

    prediction = self.model.predict(imgg)[0][0]
    
    return prediction

  def predict_torch(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    im = test_transforms(img).unsqueeze_(0)
    with torch.no_grad():
      y = self.model(im)
    
    return y[0]
  
  def run_age_estimation(self, img_in, file=None):
    img = None

    if img_in is not None:
      img = readb64_cv(img_in)
    else:
      if file is not None:
        img = read_img(file)
      else:
        return FaceWithAge(age=-1, x=0, y=0, w=img.shape[1], h=img.shape[0])
    
    return FaceWithAge(age=int(self.predict(img)), x=0, y=0, w=img.shape[1], h=img.shape[0])
