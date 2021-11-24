from pydantic import BaseModel
from enum import Enum
from http import HTTPStatus
from typing import List

class Face(BaseModel):
  x: float = 20
  y: float = 35
  w: int = 100
  h: int = 150
  face_probability: float = 0.9

class FaceWithAge(Face):
  age: int = 23

class AgeEstimationResponse(BaseModel):
  faces: List[FaceWithAge]
  message: str = HTTPStatus.OK.phrase
  status: int = HTTPStatus.OK
  
class FaceDetectionResponse(BaseModel):
  faces: List[Face]
  message: str = HTTPStatus.OK.phrase
  status: int = HTTPStatus.OK  

class FaceDetectionModels(Enum):
  yolov5s = "yolov5s"

class EstimationModels(Enum):
  effnetv1_b0 = "EfficientNet B0"
  effnetv2_b0_torch = "EfficientNetV2 B0"
  
class ListModels(Enum):
  age = "age"
  face = "face"
  all = "all"  