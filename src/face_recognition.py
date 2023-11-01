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

dfs = DeepFace.find(img_path = "../test/haland.jpg", db_path = "/home/truong/GitWorkspace/FaceID/data", model_name = models[1])

print(dfs)
print(len(dfs))
print(type(dfs))
print(type(dfs[0]))
print(dfs[0].columns)
