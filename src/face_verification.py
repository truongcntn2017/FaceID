from deepface import DeepFace

result = DeepFace.verify(img1_path = "../test/vuong.jpg", img2_path = "../data/vuong.jpg")

print(result)
