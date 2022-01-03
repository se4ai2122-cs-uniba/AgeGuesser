from http import HTTPStatus
from fastapi.testclient import TestClient
from app.main import app, estimation_models
import pytest
import json
from app.model_loader import EstimationModel, readb64_cv

@pytest.fixture
def image_file():
    return {
        'file': ('img.jpg', open('img.jpg', 'rb')),
    }

@pytest.fixture
def wrong_image_file():
    return {
        'file': ('img_base64.jpg', open('img_base64.jpg', 'rb')),
    }

@pytest.fixture
def image_base64():
    with open("img_base64.jpg", "r") as f:
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

    @pytest.mark.parametrize("model_name", ["effnetv1_b0", "effnetv2_b0_torch"])
    def test_base64_predict(model_name, image_base64, ):

        model : EstimationModel = estimation_models[model_name]

        result = model.run_age_estimation(image_base64, None)

        assert len(result["faces"]) == 1
    
    @pytest.mark.parametrize("model_name", ["effnetv1_b0", "effnetv2_b0_torch"])
    def test_base64_predict_wrong(model_name, ):

        model : EstimationModel = estimation_models[model_name]

        result = model.run_age_estimation("somedummybase64", None)

        assert len(result["faces"]) == 0

    @pytest.mark.parametrize("model_name", ["effnetv1_b0", "effnetv2_b0_torch"])
    def test_file_predict(model_name):

        model : EstimationModel = estimation_models[model_name]

        with open('img.jpg', 'rb') as f:
            result = model.run_age_estimation(None, f.read())

        assert len(result["faces"]) == 1
    
    @pytest.mark.parametrize("model_name", ["effnetv1_b0", "effnetv2_b0_torch"])
    def test_file_predict_wrong(model_name):

        model : EstimationModel = estimation_models[model_name]

        with open('img_base64.jpg', 'rb') as f:
            result = model.run_age_estimation(None, f.read())

        assert len(result["faces"]) == 0

    def test_index():
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.json()["message"] == "OK"
        assert response.json()["method"] == "GET"    
        assert response.json()["data"] == {"message":"Welcome to AgeGuesser! Please, read the `/docs`!"}
        
    @pytest.mark.parametrize("type", ["age", "face", "all"])
    def test_models_age_list(type):
        
        response = client.get("/models.%s.list"%type)

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
    
    def test_model_age_predict_invalid_imgfile(wrong_image_file, default_age_predict_payload):

        response = client.post("/models.age.predict", data=default_age_predict_payload, files=wrong_image_file )
        
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
    
    def test_model_face_predict_file_threshold_not_le(image_file, default_face_predict_payload):

        default_face_predict_payload["threshold"] = 1.1
        response = client.post("/models.face.predict", data=default_face_predict_payload, files=image_file)
        
        json_ = response.json()
        
        assert response.status_code == 422
        assert json_["detail"][0]["type"] == "value_error.number.not_le"
    
    def test_model_face_predict_file_threshold_not_ge(image_file, default_face_predict_payload):

        default_face_predict_payload["threshold"] = -0.9
        response = client.post("/models.face.predict", data=default_face_predict_payload, files=image_file)
        
        json_ = response.json()
        
        assert response.status_code == 422
        assert json_["detail"][0]["type"] == "value_error.number.not_ge"
    
