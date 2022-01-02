# fastapi backend
from datetime import datetime
from functools import wraps
from http import HTTPStatus
import json
from copy import deepcopy

#test change username
from app.model_loader import DetectionModel, EstimationModel, load_detection_model
from app.schemas import AgeEstimationResponse, EstimationModels,FaceDetectionModels,ListModels, FaceDetectionResponse
from fastapi import FastAPI, Request, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.datastructures import UploadFile


def load_models_info():
  with open("release_models/models.json", "r") as f:
    return json.load(f)


models_info = load_models_info()

# loaded models
estimation_models = {}
detection_models = {}

estimation_models_info = {}
detection_models_info = {}
all_models_info = {}
# Define application
app = FastAPI(
    title="AgeGuesser API",
    description="Deep learning based age estimation system.<br>\
      <a href='https://github.com/se4ai2122-cs-uniba/AgeGuesser'>Github</a>",
    version="0.1",
)

origins = ["http://localhost:5000","http://localhost","https://ageguesser.com", ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
  global estimation_models_info
  global detection_models_info
  global all_models_info

  estimation_models_ = models_info["age_estimation"]
  detection_models_ = models_info["face_detection"]

  for key in estimation_models_:
    model_info = estimation_models_[key]
    estimation_models[key] = EstimationModel(model_info["file_name"]) # load_estimation_model(model_info["file_name"])
  
  for key in detection_models_:
    model_info = detection_models_[key]
    detection_models[key] = load_detection_model(model_info["file_name"])
  
  models = deepcopy(models_info)
  models_ = models["age_estimation"]
  for key in models_:
    del models_[key]["file_name"]
  
  estimation_models_info = deepcopy(models_)

  models_ = models["face_detection"]
  for key in models_:
    del models_[key]["file_name"]
  
  detection_models_info = deepcopy(models_)

  all_models_info = deepcopy(estimation_models_info)
  all_models_info.update(detection_models_info)


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


@app.get("/models.{type}.list", tags=["Models"], summary="List available age estimation models.")
@construct_response
def _get_models_list_estimation(request: Request,type: ListModels ):
      

    if type.name =="age":  
    
      models_ = estimation_models_info
        
    elif type.name =="face":
          
      models_ = detection_models_info
           
    elif type.name == "all":
          
      models_ = all_models_info
      
    else:
      return {
        "message": "Type must be either \"age\" or \"face\"",
        "status-code": HTTPStatus.BAD_REQUEST
      } 
          

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": models_,
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
    <li>If both <b>file</b> and <b>img_base64</b> are set, <b>only</b> the former will be taken into account.</li>\
    <li>The <b>orientation</b> field represents the input image orientation, according to the Exif standard. It supports the following rotation values: 0 (0°), 6 (90°) and 8 (270°).\
      For more info check this <a target='_blank' href='https://sirv.com/help/articles/rotate-photos-to-be-upright/'>link</a></li>  \
      </ul>",
    
  response_model=AgeEstimationResponse, ) 
async def _post_models_age_predict(
  file: UploadFile = File(None, description="Image"), 
  model: str = Form("effnetv1_b0", description="Model ID"), 
  img_base64: str = Form(None, description="A base64 encoded image."), 
  orientation: int = Form(0, description="Image orientation according to the Exif standard. Supported values: 0 (0°), 6 (90°) and 8 (270°)."), 
  extract_faces: bool = Form(True, description="Extract face(s) before running the age prediction.") ):

    if model not in estimation_models:
      return AgeEstimationResponse(
        faces=[],
        message="Unknown model. Please look at the available ones at /models.age.list",
        status=HTTPStatus.BAD_REQUEST
        )
    
    # Assure image-file has priority 
    if file != None:
      img_base64 = None
    else:
      file = None

    if file == None and img_base64 == None:
      return AgeEstimationResponse(
        faces=[],
        message="Please upload an image file or a base64-encoded one.",
        status=HTTPStatus.BAD_REQUEST
        )
    
    #print(model.name)
    model : EstimationModel = estimation_models[model]
    
    file_ = None
    if file is not None:
      file_ = await file.read()

    if extract_faces:
      detection_model : DetectionModel = detection_models["yolov5s"]
      prediction_result = detection_model.run_prediction_with_age(model, img_base64, file_, orientation)
    else:
      prediction_result = model.run_age_estimation(img_base64, file_)
    
    return AgeEstimationResponse(
      faces=prediction_result["faces"],
      message=prediction_result["msg"],
      status=prediction_result["status"]
      )

@app.post("/models.face.predict", tags=["Prediction"], 
summary="Detect faces in an image.",
response_description="Bounding box data for each face that is detected: \
  <br><ul>\
    <li>x: x coordinate of the top-left corner</li>\
    <li>y: y coordinate of the top-left corner</li>\
    <li>w: width</li>\
    <li>h: height</li>\
    <li>face_probability: confidence of the model for the face detected</li>\
  </ul>",
response_model=FaceDetectionResponse)
async def _post_models_face_predict(
  file: UploadFile = File(None, description="Image"), 
  model: str = Form("yolov5s", description="Model ID"), 
  img_base64: str = Form(None, description="A base64-encoded image."), 
  threshold: float = Form(0.6, ge=0.0, le=1.0, description="Confidence threshold for face detection.") ):
    
    if model not in detection_models:
      return FaceDetectionResponse(
        faces=[],
        message="Unknown model. Please look at the available ones at /models.face.list",
        status=HTTPStatus.BAD_REQUEST
        )

    detection_model : DetectionModel = detection_models[model] 
    
    file_ = None
    if file is not None:
      file_ = await file.read()

    prediction_result = detection_model.run_prediction(img_base64, file_, threshold)
    return FaceDetectionResponse(
        faces=prediction_result["faces"],
        message=prediction_result["msg"],
        status=prediction_result["status"]
        )
