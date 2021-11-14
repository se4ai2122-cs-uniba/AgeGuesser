from pydantic import BaseModel

class Face(BaseModel):
  x: float = 20
  y: float = 35
  w: int = 100
  h: int = 150
  face_probability: float = 0.9

class FaceWithAge(Face):
  age: int = 23
