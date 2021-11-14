import cv2
import time
import os

def getListOfFiles(dirPath):
    listOfFiles = os.listdir(dirPath)
    allFiles = list()

    for entry in listOfFiles:
        fullPath = os.path.join(dirPath, entry)
        if os.path.isdir(fullPath):
            allFiles=allFiles+getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def main():
    dirname = 'pictures'
    listOffiles = getListOfFiles(dirname)

    for i in range(20):
        imagePath = listOffiles
        print(imagePath)
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade=cv2.CascadeClassifier(cascPath)
        image=cv2.imread(imagePath)
        gray = cv2.cvColor(image,cv2.COLOR_BGR2GRAY)

        faces=faceCascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30,30))
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x.y),(x+w, y+h), (0,255,0),2)

        cv2.imshow("faces found", image)
        cv2.waitKey(4)
        time.sleep(5)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
