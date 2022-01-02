from http import HTTPStatus
from fastapi.testclient import TestClient
from app.main import app
import pytest

@pytest.fixture
def image_file():
    return {
        'file': ('img.jpg', open('img.jpg', 'rb')),
    }

@pytest.fixture
def image_base64():
    with open("img_base64.jpg", "rb") as f:
        return f.read()


@pytest.fixture()
def default_age_predict_payload():

    payload = {
                "model": "effnetv1_b0", 
                "orientation": 0,
                "extract_faces": True,
                "img_base64": None
            }
    return payload

@pytest.fixture()
def default_face_predict_payload():

    payload = {
                "model": "yolov5s", 
                "img_base64": None,
                "threshold": 0.6
            }
    return payload

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

        for model in json_["data"]:
            model_info = json_["data"][model]
            assert model_info.get("name") is not None
            assert model_info.get("backbone") is not None
            assert model_info.get("info") is not None

    def test_model_age_predict(image_file, default_age_predict_payload):
        
        response = client.post("/models.age.predict", data=default_age_predict_payload, files=image_file)
        
        json_ = response.json()
        
        assert response.status_code == 200
        assert len(json_["faces"]) == 6
    
    def test_model_age_predict_base64(image_base64, default_age_predict_payload):

        default_age_predict_payload["img_base64"] = image_base64

        response = client.post("/models.age.predict", data=default_age_predict_payload, )
        
        json_ = response.json()
        
        assert response.status_code == 200
        assert len(json_["faces"]) == 6
    
    def test_model_age_predict_base64_no_extract(image_base64, default_age_predict_payload):

        default_age_predict_payload["img_base64"] = image_base64
        default_age_predict_payload["extract_faces"] = False
        response = client.post("/models.age.predict", data=default_age_predict_payload, )
        
        json_ = response.json()
        
        assert response.status_code == 200
        assert len(json_["faces"]) == 1
    
    def test_model_age_predict_invalid_base64(default_age_predict_payload):

        image_base64 = "somedummybase64"

        default_age_predict_payload["img_base64"] = image_base64

        response = client.post("/models.age.predict", data=default_age_predict_payload, )
        
        json_ = response.json()
        assert response.status_code == 200
        assert len(json_["faces"]) == 0
        assert "invalid" in json_["message"].lower()
        assert json_["status"] == HTTPStatus.BAD_REQUEST
    
    def test_model_wrong_model(default_age_predict_payload):

        default_age_predict_payload["model"] = "effnetv1_bO_with_typo"

        response = client.post("/models.age.predict", data=default_age_predict_payload, )
        
        json_ = response.json()
        
        assert json_["status"] == HTTPStatus.BAD_REQUEST
        assert len(json_["faces"]) == 0
        assert json_["message"] =="Unknown model. Please look at the available ones at /models.age.list"
    
    def test_model_face_predict_base64(image_base64, default_face_predict_payload):

        default_face_predict_payload["img_base64"] = image_base64

        response = client.post("/models.face.predict", data=default_face_predict_payload, )
        
        json_ = response.json()
        
        assert response.status_code == 200
        assert len(json_["faces"]) == 6
    
    def test_model_face_predict_file(image_file, default_face_predict_payload):

        response = client.post("/models.face.predict", data=default_face_predict_payload, files=image_file)
        
        json_ = response.json()
        
        assert response.status_code == 200
        assert len(json_["faces"]) == 6
    
