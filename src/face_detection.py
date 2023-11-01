from deepface import DeepFace

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

#face detection and alignment
face_objs = DeepFace.extract_faces(img_path = "../data/man_city.jpg",
        target_size = (224, 224),
        detector_backend = backends[4]
)

print(len(face_objs))
print(list(face_objs[0].keys()))
print(face_objs[0]['facial_area'])
print(face_objs[0]['confidence'])
