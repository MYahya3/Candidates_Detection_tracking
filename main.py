import cv2
from id_tracker import *
from yolov5_detection import pre_process, post_process

def tracking_id(bbox_list, frame):
	tracker = Tracker()
	box_ids = tracker.update(bbox_list)
	for box_id in box_ids:
		x1,y1,x2,y2, id = box_id
		cv2.rectangle(frame, (x1,y1), (x2,y2), (255,158,59), 2)
		cv2.rectangle(frame, (x1, y1), (x1 + 35, y1 - 35), (100,185,0), cv2.FILLED)
		cv2.putText(frame, str(id), (x1, y1) , cv2.FONT_HERSHEY_PLAIN,2, (0,0,0), 1)

	return frame


cand_remain = 0
cap = cv2.VideoCapture("demo_video/testing.mp4")
while True:
	ret, frame = cap.read()
	roi = frame[5:719, 4:980]

	# print(frame.shape)
	# Give the weight files to the model and load the network using them.
	modelWeights = "yolov5/yolov5s.onnx"
	net = cv2.dnn.readNet(modelWeights)

	# Process image.
	detections = pre_process(roi, net)
	bbox_list = post_process(roi, detections)

	cand_remain = len(bbox_list)

	# Display object count on the frame
	cv2.putText(frame, f'Candidates Remain: {cand_remain}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	img = tracking_id(bbox_list=bbox_list, frame=frame)

	cv2.imshow('Output', frame)
	key = cv2.waitKey(1)
	if key == ord("q"):
		break
cap.release()
cv2.destroyAllWindows()
