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

class EstimationModels(Enum):
  effnetv1_b0 = "age_est_1"
  effnetv2_b0_torch = "age_est_2"