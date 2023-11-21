from deepface import DeepFace


def represent(img_path, model_name, detector_backend, enforce_detection, align):
    result = {}
    embedding_objs = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = embedding_objs
    return result


def verify(
    img1_path, img2_path, model_name, detector_backend, distance_metric, enforce_detection, align
):
    obj = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )
    return obj


def analyze(img_path, actions, detector_backend, enforce_detection, align):
    result = {}
    demographies = DeepFace.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = demographies
    return result
    
def regconition(img_path, db_path):
    models = [
  	"VGG-Face",
  	"Facenet",
  	"Facenet512",
  	"OpenFace",
  	"DeepFace",
  	"DeepID",
  	"ArcFace",
  	"Dlib",
  	"SFace",
	]

    result = {}
    dfs = DeepFace.find(img_path = img_path, db_path = db_path , model_name = models[1])
    result["results"] = dfs
    return result
    
def detection(img_path):
    backends = [
  	'opencv',
  	'ssd',
  	'dlib',
  	'mtcnn',
  	'retinaface',
  	'mediapipe',
  	'yolov8',
  	'yunet',
	]	

    result = {}
    face_objs = DeepFace.extract_faces(img_path = img_path,
        target_size = (224, 224),
        detector_backend = backends[4]
    )
    result["results"] = face_objs
    return result
  
