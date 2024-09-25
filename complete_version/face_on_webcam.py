import numpy as np
import cv2
import gc
import psutil
import os



cap = cv2.VideoCapture(0) # Open the webcam (0 is the default camera)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
	
frame_count = 0
no_face_count = 0
no_face_threshold = 150

prototxt_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
confidence_threshold = 0.5
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

while True:
    # Capture frame-by-frame
    ret, frame = cap.read() #ret is a boolean, true for when frame is successfully captured, false when there is an error (eg. disconnected webcam)
    if not ret:
        print("Error: Could not read frame.")
        break
    
    frame_count +=1
    if frame_count % 100 == 0:
        # gc.collect()
        print(f"Garbage collector: collected {gc.collect()} objects.")
        print(f"Memory usage: {memory_usage():.2f} MB\n")

    #mirror the frame
    frame = cv2.flip(frame, 1) #0-vertical, 1-horizontal, -1-both

    face_detected = False #flag to determine whether there is a face or not

    #face detection through the network
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:

            face_detected = True #flag set to true

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")


            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
    if face_detected == True:
        no_face_count = 0
    else:
        no_face_count+=1


    if no_face_count == no_face_threshold:
        break
        
        
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #checks for "q" every 1 millisecond
        break



# clean up
cap.release()  #releases the webcam
cv2.destroyAllWindows()  #destroys windows, even when in background