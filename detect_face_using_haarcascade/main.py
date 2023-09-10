import cv2
cascade_classifier = cv2.CascadeClassifier('haar-cascade-files-master/haarcascade_frontalface_alt2.xml')
cam = cv2.VideoCapture(0)
while True:
    grabbed, frame = cam.read()
    if not grabbed:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_face = cascade_classifier.detectMultiScale(gray_frame,scaleFactor=1.2, minNeighbors=10)
    for x, y, w, h in detected_face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
    cv2.imshow('camera',frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
cam.release()
cv2.destroyAllWindows()
