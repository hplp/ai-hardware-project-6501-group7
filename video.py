import cv2
from predict import *
# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Write the frame to the output file
    out.write(frame)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Save the frame as image.png when 't' is pressed
    if cv2.waitKey(1) == ord('t'):
        cv2.imwrite('img.png', frame)
        print("Captured frame saved as image.png")
        start_test()

    # Press 's' to exit the loop
    if cv2.waitKey(1) == ord('s'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
