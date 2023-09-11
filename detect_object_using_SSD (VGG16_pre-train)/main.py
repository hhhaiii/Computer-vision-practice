import cv2
import numpy as np
# concern about object have confidence > 50%
THRESHOLD = 0.5
# Non-max_suppression (IOU)
THRESHOLD_NMS = 0.4
with open('classes.txt', 'r') as file:
    content = file.read()
values = content.split(',')
COCO_DATASET = [value.strip() for value in values]

def show_image_predict(image, NMS_id, classes_id, confident, bboxes):
    # Non-max-suppression
    for index in NMS_id:
        box = bboxes[index]
        x, y, w, h = box[0:4]
        if classes_id[index] > 0 and classes_id[index] <= len(COCO_DATASET):
            cv2.rectangle(image, (x, y), (x + w, y + h), 2)
            title = COCO_DATASET[classes_id[index]-1] + ': %.2f' % (confident[index])
            cv2.putText(image, text=title, org=(x, y - 5), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=0.5, color=(255, 0, 0), thickness=2)


camera = cv2.VideoCapture(0)
neural_network = cv2.dnn.DetectionModel('frozen_inference_graph.pb', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
neural_network.setInputSize(480, 640)
# Convert value to range [-1,1]
neural_network.setInputScale(1.0/255)
neural_network.setInputMean(127.5)
neural_network.setInputSwapRB(True)

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break
    classes_id, confident, bboxes = neural_network.detect(frame)
    bboxes = [list(box) for box in bboxes]
    confident = list(confident)
    classes_id = list(classes_id)
    NMS_id = cv2.dnn.NMSBoxes(bboxes, confident, THRESHOLD, THRESHOLD_NMS)
    show_image_predict(frame, NMS_id, classes_id, confident, bboxes)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
camera.release()
cv2.destroyAllWindows()