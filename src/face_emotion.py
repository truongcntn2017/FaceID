from deepface import DeepFace

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

objs = DeepFace.analyze(img_path = "../test/vuong.jpg", actions=['emotion'])

print(len(objs))
print(objs[0]['dominant_emotion'])

