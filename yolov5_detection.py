import cv2
import numpy as np

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.50
CONFIDENCE_THRESHOLD = 0.50

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
THICKNESS = 1

# Colors
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

def pre_process(input_image, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
	# Sets the input to the network.
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)
	# print(outputs[0].shape)

	return outputs


def post_process(input_image, outputs):
	# Lists to hold respective values while unwrapping.
	class_ids = []
	confidences = []
	boxes = []
	bbox = []
	# Rows.
	rows = outputs[0].shape[1]


	image_height, image_width = input_image.shape[:2]

	# Resizing factor.
	x_factor = image_width / INPUT_WIDTH
	y_factor = image_height / INPUT_HEIGHT

	# Iterate through 25200 detections.

	for r in range(rows):
		row = outputs[0][0][r]
		confidence = row[4]

		# Discard bad detections and continue.
		if confidence >= CONFIDENCE_THRESHOLD:
			classes_scores = row[5:]

			# Get the index of max class score.
			class_id = np.argmax(classes_scores)
			#  Continue if the class score is above threshold.
			if (classes_scores[class_id] > SCORE_THRESHOLD) and class_id == 0:
				confidences.append(confidence)
				class_ids.append(class_id)

				cx, cy, w, h = row[0], row[1], row[2], row[3]

				left = int((cx - w / 2) * x_factor)
				top = int((cy - h / 2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)

				box = np.array([left, top, width, height])
				boxes.append(box)

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		x1,y1,x2,y2 =left, top, left + width, top + height
		bbox.append([x1,y1,x2,y2])
	return bbox