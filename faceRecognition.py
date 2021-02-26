import cv2, numpy, os
import urllib.request
import imutils
# getting haarcascade algorithm to detect the face
haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
# loading dataset folder
datasets = 'datasets'
print('Training...')

(images, labels, names, id) = ([], [], {}, 0)
# inside the dataset folder
for (subdirs, dirs, files) in os.walk(datasets):
    # inside subdirectories assign id's to each each
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id +=1
# images and lables in a numpy array
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
print(images, labels)                   
(width, height) = (130, 100)
# initialize classifier
model = cv2.face.LBPHFaceRecognizer_create()
# we can use this too,just comment the above one
#model =  cv2.face.FisherFaceRecognizer_create()

# training the model,this classifier takes less time than deep learning
model.train(images, labels)

# webcam = cv2.VideoCapture(0)
cnt=0
url='http://192.168.29.147:8080/shot.jpg'
while True:
    # (_, im) = webcam.read()
    imgPath = urllib.request.urlopen(url)
    imgNp = numpy.array(bytearray(imgPath.read()), dtype=numpy.uint8)
    im = cv2.imdecode(imgNp, -1)
    im = imutils.resize(im, width=450)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # prediction from trained model
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # checking the prediction
        if prediction[1]<800:
            # printing text
            cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,2,(0, 0, 255))
            print (names[prediction[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putText(im,'Unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            if(cnt>100):
                print("Unknown Person")
                # saving the image of unknown person
                cv2.imwrite("unKnown.jpg",im)
                cnt=0
    cv2.imshow('FaceRecognition', im)
    key = cv2.waitKey(10)
    # esc to quit
    if key == 27:
        break

# webcam.release()
cv2.destroyAllWindows()



