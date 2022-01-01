# Andrew Maney 1-1-2022

# Imports
import cv2
from imutils.video import VideoStream  
import time

# Print helpful information
print("Face Detector Example\nWritten by Andrew Maney in 2022.\nPress the escape key to exit.\n")

# Haarcascade classifier file for detecting faces. This tells python exactly how it should detect faces
face_cascade = cv2.CascadeClassifier('face.xml')

# Use the Raspberry Pi's Camera (If Present)
PiCamera = True

# Set initial frame size and text location.
sizeX = 320
sizeY = 240
textX = int(sizeX/3)-25
textY = int(sizeY/18)+10
frameSize = (sizeX, sizeY)

# Setup video stream
print("[INFO] >> Initializing Camera...")
vs = VideoStream(src=0, usePiCamera=PiCamera, resolution=frameSize,framerate=32).start()

# Allow camera to initialize.
time.sleep(0.5)
i = 0
print("[DONE]\n\n[INFO] >> Searching for faces...")


while True:
    # Read Video steram
    img = vs.read()
    
    # Convert the frame into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find faces in frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        # Show a message if there are no detected faces        
        cv2.putText(img, "NO FACE DETECTED", (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,cv2.LINE_AA)
    else:
        # Show a message if there are one or more detected faces
        cv2.putText(img, "FACE DETECTED", (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,cv2.LINE_AA)
        for (x,y,w,h) in faces:
            # Draw a rectangle around every face that is detected
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            i = i+1            
            # Output a message to the console when a face is detected    
            print("[INFO] >> Faces were found {} times. Face Position: X={} || Y={}".format(i,x,y))
            
    # Display the result
    cv2.imshow('Face Detection',img)
    
    # Exit the program when the user presses the escape key
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print("[INFO] >> User requested program exit.")
        break
    
    #Sleep statement to help reduce CPU usage
    time.sleep(0.02)

#Close all windows when the user presses escape
print("[INFO] >> Exitting...")
cv2.destroyAllWindows()
print("[DONE]")


