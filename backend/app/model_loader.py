from http import HTTPStatus
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
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
import timm
from torch import nn
from utils.torch_utils import select_device


img_height = 224
img_width = 224

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    transforms.transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    torchvision.transforms.Resize((img_height, img_height)),
])

def readb64_cv(base64_):
    msg = base64.decodebytes(bytes(base64_.split(",")[1], "utf-8"))
    buf = io.BytesIO(msg)
    pil_image = Image.open(buf).convert('RGB') 
    # pil_image.save('my-image.jpeg')
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def read_img(img_file, orientation=0):
    buf = io.BytesIO(img_file)
    pil_image = Image.open(buf).convert('RGB')
    deg = {3:180,6:270,8:90}.get(orientation, 0)

    if deg != 0:
        pil_image=pil_image.rotate(deg, expand=True)
    # pil_image.save('my-image.jpeg')
    open_cv_image = np.array(pil_image) 
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)


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
            
      def forward(self, x):
          y_ = self.backbone(x)
          y_sup = self.model1(y_)

          return y_sup
    resnet = timm.create_model('tf_efficientnetv2_b0', pretrained=False)

    backbone = nn.Sequential(
        *list(resnet.children())[:-1]
    )

    model = AgeNetwork(backbone).to(torch.device('cpu'))
    
    model.load_state_dict( torch.load(self.weights_path, map_location=torch.device('cpu'))["model_parameters"])
    model.eval() # important
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
      try:
        img = readb64_cv(img_in)
      except:
        return { "faces" : [], "status": HTTPStatus.BAD_REQUEST, "msg": "Invalid base64-encoded image."}
    else:
      try:
        img = read_img(file, )
      except:
        return { "faces" : [], "status": HTTPStatus.BAD_REQUEST, "msg": "Invalid image."}
    
    return { "faces" : [FaceWithAge(age=int(self.predict(img)), x=0, y=0, w=img.shape[1], h=img.shape[0]) ],
              "status": HTTPStatus.OK, "msg": HTTPStatus.OK.phrase }


class DetectionModel(object):
    
  def __init__(self, model, stride, device, imgsz,) -> None:
    super().__init__()
    self.model = model
    self.stride = stride
    self.device = device
    self.imgsz = imgsz
  
  def readb64_cv(self, base64_):
    msg = base64.decodebytes(bytes(base64_, "utf-8"))
    buf = io.BytesIO(msg)
    pil_image = Image.open(buf).convert('RGB') 
    # pil_image.save('my-image.jpeg')
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    img = open_cv_image[:, :, ::-1].copy() 
    
    return img

  def setup_img(self, open_cv_image):
    #open_cv_image = np.array(pil_image)
    dst = letterbox(open_cv_image, new_shape=(320, 320), stride=self.stride)[0]
    
    dst = dst.transpose(2, 0, 1)
    dst = np.ascontiguousarray(dst)

    return dst

  def predict(self, img, im0, threshold,):

    img = torch.from_numpy(img).to(self.device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = self.model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, threshold, 0.45, agnostic=False)

    boxess = []

    # print(im0.shape)
    for i, det in enumerate(pred):  # detections per image 

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):

                f = Face(x=int(xyxy[0]),y=int(xyxy[1]),w=int(xyxy[2]-xyxy[0]), h=int(xyxy[3]-xyxy[1]), face_probability=float("%.2f" % conf.item()))
                boxess.append(f)

    return boxess

  def predict_with_age(self, img, im0, threshold, age_model : EstimationModel):

    img = torch.from_numpy(img).to(self.device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = self.model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, threshold, 0.45, agnostic=False)

    boxess = []

    # print(im0.shape)
    for i, det in enumerate(pred):  # detections per image 

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):

                face = im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                # cv2.imwrite("lol.jpg", face)
                # run age classifier
                age = age_model.predict(face) # age_estimation(age_model, face)
                
                f = FaceWithAge(x=int(xyxy[0]),y=int(xyxy[1]),w=int(xyxy[2]-xyxy[0]), h=int(xyxy[3]-xyxy[1]), age=int(age), face_probability=float("%.2f" % conf.item()))# Rect(int(xyxy[0]),int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1]), int(age), "%.2f" %  conf.item())

                boxess.append(f)

                # cv2.imwrite("f.jpg", face)
                # cv2.waitKey(1)

    return boxess #[b.data() for b in boxess]

  def run_prediction(self, img_base64, img_file, threshold=0.6):

    if img_base64 is not None:
      try:
        im0 = self.readb64_cv(img_base64.split(",")[1])
      except:
        return { "faces" : [], "status": HTTPStatus.BAD_REQUEST, "msg": "Invalid base64-encoded image."}
    else:
      try:
        im0 = read_img(img_file)
      except:
        return { "faces" : [], "status": HTTPStatus.BAD_REQUEST, "msg": "Invalid image."}


    img = self.setup_img(im0) # img ready for yolo
    return { "faces": self.predict(img, im0, threshold), "status": HTTPStatus.OK, "msg": HTTPStatus.OK.phrase}
  
  def run_prediction_with_age(self, age_model, img_base64, file_img, orientation, threshold=0.6,):

    if img_base64 is not None:
      try:
        im0 = self.readb64_cv(img_base64.split(",")[1]) # orig image
      except:
        return { "faces" : [], "status": HTTPStatus.BAD_REQUEST, "msg": "Invalid base64-encoded image."}
    else:
      try:
        im0 = read_img(file_img, orientation)
      except:
        return { "faces" : [], "status": HTTPStatus.BAD_REQUEST, "msg": "Invalid image."}
    
    img = self.setup_img(im0) # img ready for yolo

    return { "faces": self.predict_with_age(img, im0, threshold, age_model), "status": HTTPStatus.OK, "msg": HTTPStatus.OK.phrase}


def load_detection_model(weights_path):
	imgsz = 320
	device = select_device("cpu")

	# Load model
	model = attempt_load(weights_path, map_location=device)  # load FP32 model
	stride = int(model.stride.max())  # model stride
	imgsz = check_img_size(imgsz, s=stride)  # check img_size

	return DetectionModel(model, stride, device, imgsz)