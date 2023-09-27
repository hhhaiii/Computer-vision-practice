import cv2
import numpy as np
YOLO_SIZE = 320
# just detect object > 30% probability
THRESHOLD = 0.3
# Non-Max suppression (IOU) > 30%
THRESHOLD_NMS = 0.3

# Read classes and push into array
with open('classes.txt', 'r') as file:
    content = file.read()
values = content.split(',')
COCO_DATASET = [value.strip() for value in values]



def detect_object(outputs):
    # output have size = (n,85) with n is number of bounding box and 85 is predict vector
    # 85 = [x, y, w, h, confident] + 80 class in COCO dataset
    bboxs = []
    target_ids = []
    confidents = []
    for output in outputs:
        for predict in output:
            if np.max(predict[5:]) > THRESHOLD:
                x, y, w, h = predict[0:4]*YOLO_SIZE
                # Covert x,y to conner (now it's a center)
                x = int(x - w/2)
                y = int(y-h/2)
                target_ids.append(np.argmax(predict[5:]))
                confidents.append(np.max(predict[5:]))
                bboxs.append([x, y, w, h])
    print(confidents)
    bbox_NMS_id = cv2.dnn.NMSBoxes(bboxes=bboxs, scores=confidents, score_threshold=THRESHOLD, nms_threshold=THRESHOLD_NMS)
    return bbox_NMS_id, bboxs, target_ids, confidents


def show_image_with_bbox(image, bbox_NMS_id, bboxs, target_ids, confidents, scale_width, scale_height):
    for index in bbox_NMS_id:
        bbox = bboxs[index]
        x, y, w, h = bbox[0:4]
        # Transform to original image
        x = int(x*scale_width)
        y = int(y*scale_height)
        w = int(w*scale_width)
        h = int(h*scale_height)
        title = COCO_DATASET[target_ids[index]] + ': %.2f' % confidents[index]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, text=title, org=(x, y-10), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=0.5, color=(255, 0, 0), thickness=2)


def detect_frame(frame):
    # Turn image into a BLOB
    original_width = frame.shape[1]
    original_height = frame.shape[0]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(YOLO_SIZE, YOLO_SIZE), swapRB=True, crop=False)
    neural_network.setInput(blob)
    layer_names = neural_network.getLayerNames()
    # get value from 3 output layer
    output_names = [layer_names[index - 1] for index in neural_network.getUnconnectedOutLayers()]
    outputs = neural_network.forward(output_names)
    final_bbox, all_bbox, target_ids, confident = detect_object(outputs)
    show_image_with_bbox(frame, final_bbox, all_bbox, target_ids, confident, original_width/YOLO_SIZE, original_height/YOLO_SIZE)
    cv2.imshow('window', frame)

video = cv2.VideoCapture('videoplayback1.mp4')
# Using pre-train YOLO model
neural_network = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# Configure the backend and processing target for the neural network (CPU)
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
while True:
    grabbed, frame = video.read()
    if not grabbed:
        break
    detect_frame(frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

video.release()
cv2.destroyAllWindows()