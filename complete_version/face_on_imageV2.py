import numpy as np
import cv2


#TO DO:
#resize the output picture
#modify argument line, i.e. hard-code it
#face ID


# Hardcoded file paths and parameters
image_path = "Several.jpeg"  #change the jpeg if needed 
prototxt_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
confidence_threshold = 0.5

# Load the serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(image_path)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(0, detections.shape[2]):
	# Extract the confidence (i.e., probability) associated with the prediction
	confidence = detections[0, 0, i, 2]

	# Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
	if confidence > confidence_threshold:
		# Compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# Draw the bounding box of the face along with the associated probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


scale_percent = 50  # percent of original size, adjust this value
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(image, (width, height)) 


# Show the output image
cv2.imshow("Output", resized_image)
cv2.waitKey(0) # Input 0 to show output for an indefinite amount of time
