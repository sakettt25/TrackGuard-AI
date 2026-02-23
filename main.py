
import os
import random
import numpy as np
from deep_sort1.deep_sort import DeepSort
from deep_sort1.sort.tracker import Tracker
# from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from ultralytics import YOLO
import threading
import time
import datetime
import clx.xms
import requests

###########################
# By creating an account in sinch sms You can get your code.
# code for sms starts here
# client is a object that carries your unique token.
# client = clx.xms.Client(service_plan_id='fea33ef771b74fee84fcf43c7e803de6',
#                         token='b65dbe1cbe8e4f0d85d7979f1cc7b0c1')
#
# create = clx.xms.api.MtBatchTextSmsCreate()
# create.sender = "447520651418"
# create.recipients = {"917489386756"}
# create.body = 'Fire detected in cam 1'
#####################
#CWD = os.getcwd()
# Create a list to store frame data

frame_data_list = []

video_path = os.path.join('.', 'data', 'test2.mp4')
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
frame_width = int(cap.get(3))  # 3 corresponds to CV_CAP_PROP_FRAME_WIDTH
frame_height = int(cap.get(4))  # 4 corresponds to CV_CAP_PROP_FRAME_HEIGHT

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),(frame.shape[1], frame.shape[0]))
#######################  COCO MODEL
model = YOLO("yolov8n.pt")
#######################  FIRE MODEL
modelf = YOLO('weights/best.pt')
############################
deep_sort_weight='deep_sort1/deep/checkpoint/ckpt.t7'
tracker=DeepSort(model_path=deep_sort_weight,max_age=30)
#######################
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
detection_threshold = 0.5
integer_keys = list(range(80))
values = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat","traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog","horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite","baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle","wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich","orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch","potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote","keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book","clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
class_map = {key: value for key, value in zip(integer_keys, values)}            #maps integer to class labels
color_to_label = {
    (0, 165, 255): 'Fire Personnel',
    (128, 0, 128): 'Station Personnel',
    (31, 31, 31): 'Cleaning Personnel',
    (47, 47, 47): 'Security Personnel'
}       #pink for cleaners,brown for security,purple for station personnel,orange for fire personnel
# --------------------
##########################                                                      #alarm generate
# Initialize pyttsx3 for voice alerts
# alarm_sound = pyttsx3.init('sapi5')
# voices = alarm_sound.getProperty('voices')
# alarm_sound.setProperty('voice', voices[0].id)
# alarm_sound.setProperty('rate', 150)
# crowd_alert_triggered = False
#
# def crowd_alert():
#     global crowd_alert_triggered
#     alarm_text = "Crowd detected!"
#     alarm_sound.say(alarm_text)
#     alarm_sound.runAndWait()
#     crowd_alert_triggered = True
##############################
flag=0
while ret:
    #########################################
    frame_data = {}  # Create a dictionary to store frame data
    resultsfire = modelf.predict(source=frame, conf=0.20)  # fire detection model
    bounding_boxes = resultsfire[0].boxes.xyxy  # Assuming xyxy format for bounding boxes
    confidences = resultsfire[0].boxes.conf
    class_labels = resultsfire[0].boxes.cls

    if(len(resultsfire)==0):
        frame_data["fire_detected"] = False
    for box, confidence, class_label in zip(bounding_boxes, confidences, class_labels):
        x_min, y_min, x_max, y_max = box.tolist()
        confidence = confidence.item()
        class_label = int(class_label.item())
        print(f"Fire&smoke confidence:{confidence}")
        # Calculate the center point
        x_centerf = (x_min + x_max) // 2
        y_centerf = (y_min + y_max) // 2
        # $$$$$$$$$$$$$$
        frame_data["fire_detected"] = True
        frame_data["fire_confidence"] = confidence
        # $$$$$$$$$$$$$

        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        if (class_label == 0):
            color = (255, 0, 255)
        else:                                                                               ##shayad 1 smoke ke liye hain
            color = (255, 0, 0)

        # x_centerf = int(x_centerf)
        # y_centerf = int(y_centerf)
        #
        #
        # cv2.circle(frame, (x_centerf, y_centerf - 5), 15, (0, 0, 255), -1)
        # cv2.rectangle(frame, (0, 180), (250, 250), (255, 0, 0), -1)
        # cv2.putText(frame, f"Fire", (frame_width-75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)                    #to draw fire bboxes
# -------------
        # if (flag == 0):
        #     flag=1
        #     try:
        #         batch = client.create_batch(create)
        #     except (requests.exceptions.RequestException, clx.xms.exceptions.ApiException) as ex:
        #         print('Failed to communicate with XMS: %s' % str(ex))
    #######################
    results = model(frame)

    # $$$$$$$$$$$$$$$$$$$$$$
    timestamp = time.time()
    timestamp_datetime = datetime.datetime.fromtimestamp(timestamp)
    formatted_timestamp = timestamp_datetime.strftime("%A, %d %B %Y %I:%M:%S %p")
    frame_data["timestamp"] = formatted_timestamp

    # $$$$$$$$$$$$$$$$$$$
    num_keys = 80                                                                           # number of class labels in coco
    dict = {key: 0 for key in range(num_keys)}                                              # this dict maps integer to detection class counts
    for result in results:                                                                  # results for the coco model
        bboxes_xywh=[]
        confidence=[]
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            w=x2-x1
            h=y2-y1
            bbox_xywh=[x1,y1,w,h]
            bboxes_xywh.append(bbox_xywh)
            confidence.append(score)
            class_id = int(class_id)
            dict[class_id] += 1                                                             # updates detection count of particular class

            # if score > detection_threshold:
            # detections.append([x1, y1, x2, y2, score])

        filtered_dict = {class_map[key]: value for key, value in dict.items() if value != 0} # filtered dict will only contain those pairs !=0 count
        for key, value in filtered_dict.items():
            print(f"{key}: {value}")

        tracks = tracker.update(bboxes_xywh, confidence, frame)
        personnel_detected = []
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            bbox_xywh = np.array(track.to_tlwh())                               # Convert to NumPy array

            x, y, w, h = bbox_xywh                                              # Extract x, y, width, and height

            shift_per = 0.5
            y_shift = int(h * shift_per)
            x_shift = int(w * shift_per)
            y += y_shift
            x += x_shift
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (colors[track_id % len(colors)]), 3)

            # center coordinates
            x_center = int(x + w / 2)
            y_center = int(y + h / 2)

            if 0 <= x_center < frame.shape[1] and 0 <= y_center < frame.shape[0]:
                center_color = frame[y_center, x_center]

                try:
                    # Ensure that center_color contains valid numeric values
                    center_color = tuple(map(int, center_color))  # Convert color to integer values

                    # Check if the detected color is in the color-to-label mapping
                    if center_color in color_to_label:
                        label = color_to_label[center_color]
                        personnel_detected.append(label)
                        print(f"{label} detected in cam")

                    cv2.circle(frame, (x_center, y_center - 5), 10, center_color, -1)
                except Exception as e:
                    print(f"Error processing track {track_id}: {e}")
            else:
                print(f"Center coordinates ({x_center}, {y_center}) are out of bounds.")


        #############################
            # Store the list of detected personnel labels in the frame data
            frame_data["personnel_detected"] = personnel_detected

            text_annotations = [(key, value) for key, value in filtered_dict.items()]  # maps label to their corresponding detection counts
            cv2.rectangle(frame, (0, 0), (250, 180), (222, 49, 99), -1)
            y = 60
            frame_data["class_counts"] = filtered_dict
            for key, value in text_annotations:
                cv2.putText(frame, f"{key}: {value}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y += 30                                                                 # if new label detected then increment y
                ###################                                                     #alarm sound
                # if key == 'person':
                #     crowd_threshold = 30  # Set your desired crowd threshold
                #     if value >= crowd_threshold and not crowd_alert_triggered:
                #         crowd_alert_thread = threading.Thread(target=crowd_alert)
                #         crowd_alert_thread.start()
                ##################
            # $$$$$$$$$$$$$$
            frame_data_list.append(frame_data)
            # print(frame_data_list)
            # # cv2.putText(frame,str(len(detections)),(25,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cap_out.write(frame)
    ret, frame = cap.read()
    # ----------------------

cap.release()
cap_out.release()
cv2.destroyAllWindows()
# Write frame data list to a text file
output_file_path = "frame_data.txt"
with open(output_file_path, "w") as file:
    for frame_data in frame_data_list:
        file.write(f"Frame Data:\n{frame_data}\n\n")

print(f"Frame data written to {output_file_path}")

