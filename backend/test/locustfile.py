import time
from locust import FastHttpUser, task, between

img = ""
with open("../img_base64.jpg", "r") as f:
  img = f.read()


class ListUser(FastHttpUser):

  @task(2)
  def models_age_list(self):
    self.client.get("/models.age.list")
  
  @task(1)
  def models_all_list(self):
    self.client.get("/models.all.list")


class CuriousUser(FastHttpUser):
  wait_time = between(1, 3)

  @task
  def models_age_predict(self):
    
    self.client.post(f"/models.age.predict",
      name="/predict", 
      data = {
          "model": "effnetv1_b0", 
          "img_base64": img, 
          "orientation": 0,
          "extract_faces": True
      } )

class HungryUser(FastHttpUser):
  wait_time = between(1, 3)

  @task(3)
  def models_age_predict(self):
    
    self.client.post(f"/models.age.predict",
      name="/predict", 
      data = {
          "model": "effnetv1_b0", 
          "img_base64": img, 
          "orientation": 0,
          "extract_faces": True
      } )
  
  @task(2)
  def models_age_list(self):
    self.client.get("/models.age.list")
  
  @task(2)
  def models_age_list(self):
    self.client.get("/models.all.list")