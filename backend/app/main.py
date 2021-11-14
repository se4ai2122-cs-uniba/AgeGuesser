# fastapi backend
from datetime import datetime
from functools import wraps
from http import HTTPStatus
import json
from copy import deepcopy

#test change username
from app.model_loader import EstimationModel, load_detection_model
from fastapi import FastAPI, Request


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

