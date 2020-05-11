import os 
import face_recognition

filelist = os.listdir('known')

known_names, known_images, known_encodings = [], [], []

for file in filelist:
    known_names.append(file[:-4])
    
    image = face_recognition.load_image_file('known/'+file)
    
    known_images.append(image)
    known_encodings.append(face_recognition.face_encodings(image)[0])


unknown_image = face_recognition.load_image_file("unknown/double.jpeg")
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

c, s = 0, 0
while c<len(known_names):
    if face_recognition.compare_faces([known_encodings[c]], unknown_encoding)[0]:
        print('It is '+known_names[c])
        s = 1
        
    c+=1

if(s==0):
    print('uknown person')
