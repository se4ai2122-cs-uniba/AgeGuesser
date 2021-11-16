# fastapi backend
from datetime import datetime
from functools import wraps
from http import HTTPStatus
import json
from copy import deepcopy

#test change username
from app.model_loader import DetectionModel, EstimationModel, load_detection_model
from backend.app.schemas import AgeEstimationResponse, EstimationModels
from fastapi import FastAPI, Request, File, Form
from fastapi.datastructures import UploadFile


def load_models_info():
  with open("release_models/models.json", "r") as f:
    return json.load(f)


models_info = load_models_info()

# loaded models
estimation_models = {}
detection_models = {}


# Define application
app = FastAPI(
    title="AgeGuesser API",
    description="Deep learning based age estimation system.<br>\
      <a href='https://github.com/se4ai2122-cs-uniba/AgeGuesser'>Github</a>",
    version="0.1",
)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap

@app.on_event("startup")
async def _load_models():

  estimation_models_ = models_info["age_estimation"]
  detection_models_ = models_info["face_detection"]

  for key in estimation_models_:
    model_info = estimation_models_[key]
    estimation_models[key] = EstimationModel(model_info["file_name"]) # load_estimation_model(model_info["file_name"])
  
  for key in detection_models_:
    model_info = detection_models_[key]
    detection_models[key] = load_detection_model(model_info["file_name"])




@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to AgeGuesser! Please, read the `/docs`!"},
    }
    return response


@app.get("/models.list", tags=["Models"], summary="List available models.")
@construct_response
def _get_models_list(request: Request):

    models = deepcopy(models_info)
    estimation_models_ = models["age_estimation"]
    detection_models_ = models["face_detection"]
    for key in estimation_models_:
      del estimation_models_[key]["file_name"]
    
    for key in detection_models_:
      del detection_models_[key]["file_name"]
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": models,
    }

    return response


@app.get("/models.age.list", tags=["Models"], summary="List available age estimation models.")
@construct_response
def _get_models_list_estimation(request: Request,):
    
    models = deepcopy(models_info)
    estimation_models_ = models["age_estimation"]

    for key in estimation_models_:
      del estimation_models_[key]["file_name"]

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": estimation_models_,
    }
    return response


@app.get("/models.face.list", tags=["Models"], summary="List available face detection models.")
@construct_response
def _get_models_list_detection(request: Request,):
    
    models = deepcopy(models_info)
    detection_models_ = models["face_detection"]

    for key in detection_models_:
      del detection_models_[key]["file_name"]

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": detection_models_,
    }
    return response


@app.post("/models.age.predict", tags=["Prediction"], 
  summary="Guess the age from a facial image.",
  response_description="Bounding box data for each face that is detected: \
    <br><ul>\
      <li>x: x coordinate of the top-left corner</li>\
      <li>y: y coordinate of the top-left corner</li>\
      <li>w: width</li>\
      <li>h: height</li>\
      <li>age: predicted age</li>\
      <li>face_probability: confidence of the model for the face detected</li>\
    </ul>",
  description="Run the Age Estimation model of choice on an image (uploaded as file or base64-encoded).<br> \
    <ul>\
    <li>If <b>extract_faces</b> is set to true, a face detection model will extract the faces first, then the age will be predicted for each one of them.</li>\
    <li>If both <b>file</b> and <b>img_base64</b> are set, only the latter will be taken into account.</li>\
      </ul>",
    
  response_model=AgeEstimationResponse, ) 
async def _post_models_age_predict(file: UploadFile = File(None, description="Image"), model: EstimationModels = Form("age_est_1", description="Model ID"), img_base64: str = Form(None, description="A base64 encoded image."), extract_faces: bool = Form(False, description="Extract the face(s) before running the age prediction.") ):
    
    if model.value not in estimation_models:
      return AgeEstimationResponse(
        faces=[],
        message="Unknown model. Please look at the available ones at /models.age.list",
        status=HTTPStatus.BAD_REQUEST
        )
     
    model : EstimationModel = estimation_models[model.value]
    
    file_ = None
    if file is not None:
      file_ = await file.read()

    if extract_faces:
      detection_model : DetectionModel = detection_models["face_det_1"]
      return AgeEstimationResponse(
        faces=detection_model.run_prediction_with_age(model, img_base64, file_),
        message=HTTPStatus.OK.phrase,
        status=HTTPStatus.OK
        )
    else:
      return AgeEstimationResponse(
        faces=[model.run_age_estimation(img_base64, file_)],
        message=HTTPStatus.OK.phrase,
        status=HTTPStatus.OK
        )
