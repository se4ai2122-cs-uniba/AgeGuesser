from http import HTTPStatus
import json
from fastapi.testclient import TestClient
from app.main import app


with TestClient(app) as client:

    def test_index():
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.json()["message"] == "OK"
        assert response.json()["method"] == "GET"    
        assert response.json()["data"] == {"message":"Welcome to AgeGuesser! Please, read the `/docs`!"}
        
    def test_models_age_list():
        
        response = client.get("/models.age.list")

        json_ = response.json()
         
        assert response.status_code == 200
        assert json_["message"] == "OK"
        assert json_["method"] == "GET"    
        assert json_["data"] == {"effnetv1_b0": {"name": "Age Estimation v1", "backbone": "EfficientNet-B0", "info": "MAE: 3 - STD: 2"}, "effnetv2_b0_torch": {"name": "Age Estimation v2", "backbone": "EfficientNetV2-B0", "info": "MAE: 3 - STD 1.3"}}

    def test_model_age_predict():
    
        files= {
            'file': ('img.jpg', open('img.jpg', 'rb')),
        }
        
        multipart_form_data = {
                "model": "effnetv1_b0", 
                "orientation": 0,
                "extract_faces": True
            }

        response = client.post("/models.age.predict", data=multipart_form_data, files=files)
        
        json_ = response.json()
        
        assert response.status_code == 200
        assert len(json_["faces"]) == 6
    
    def test_model_age_predict_base64():

        img = ""
        with open("img_base64.jpg", "rb") as f:
            img = f.read()

        multipart_form_data = {
                "model": "effnetv1_b0", 
                "orientation": 0,
                "extract_faces": True,
                "img_base64": img
            }

        response = client.post("/models.age.predict", data=multipart_form_data, )
        
        json_ = response.json()
        """ with open("result.json","w") as f:
            json.dump(response.json(),f) """
        assert response.status_code == 200
        assert len(json_["faces"]) == 6
    
    def test_model_wrong_model():

        img = ""
        with open("img_base64.jpg", "rb") as f:
            img = f.read()

        multipart_form_data = {
                "model": "effnetv1_bO_with_typo", 
                "orientation": 0,
                "extract_faces": True,
                "img_base64": img
            }

        response = client.post("/models.age.predict", data=multipart_form_data, )
        
        json_ = response.json()
        
        assert json_["status"] == HTTPStatus.BAD_REQUEST
        assert len(json_["faces"]) == 0
        assert json_["message"] =="Unknown model. Please look at the available ones at /models.age.list"
    
