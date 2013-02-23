import cv, cv2
 
HAAR_FACE_CASCADE_PATH = "face.xml"
HAAR_MOUTH_CASCADE_PATH = "mouth.xml"
HAAR_EYES_CASCADE_PATH = "eyes.xml"
VIDEO_PATH = "video.mov"
CAMERA_INDEX = 0

def detect(image, cascade, res):
    found = []
    detected = cv.HaarDetectObjects(image, cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING | cv.CV_HAAR_FIND_BIGGEST_OBJECT, res)
    if detected:
        for (x,y,w,h),n in detected:
            found.append((x,y,w,h))
    return found
 
if __name__ == "__main__":
    cv.NamedWindow("Video", cv.CV_WINDOW_AUTOSIZE)
 
    capture = cv.CaptureFromCAM(CAMERA_INDEX)
    storage = cv.CreateMemStorage()
 
    i = 0
    while True:
        image = cv.QueryFrame(capture)
 
        if i % 10 == 0:
            face = detect(image, cv.Load(HAAR_FACE_CASCADE_PATH), (100,100))
            if not face:
                continue
            tmpimg = image

            first_face = face[0]
            cv.SetImageROI(tmpimg, first_face)
            eyes = detect(tmpimg, cv.Load(HAAR_EYES_CASCADE_PATH), (50,50))
            mouth = detect(tmpimg, cv.Load(HAAR_MOUTH_CASCADE_PATH), (50,50))

        for (x,y,w,h) in face:
            cv.Rectangle(image, (x,y), (x+w,y+h), cv.RGB(255, 0, 0), 2)
        for (x,y,w,h) in eyes:
            cv.Rectangle(tmpimg, (x,y), (x+w,y+h), cv.RGB(0, 255, 0), 2)
        for (x,y,w,h) in mouth:
            cv.Rectangle(tmpimg, (x,y), (x+w,y+h), cv.RGB(0, 0, 255), 2)
 
        cv.ShowImage("window", image)
        i += 1
