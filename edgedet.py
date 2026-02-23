import cv2
import os



# image = cv2.imread('track.jpg')
# # Check if the image was loaded successfully
# if image is None:
#     print("Error: Could not load image.")
#     exit()

video_path = os.path.join('.', 'data', 'test.mp4')
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()


    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        exit()
    lower=int(max(0,0.07*64))
    upper=int(min(255,1.3*64))

    min_area_threshold = 550
    blurred_frame = cv2.GaussianBlur(frame, (1, 1), 0)
    edges = cv2.Canny(blurred_frame, threshold1=127, threshold2=127)  # You can adjust the threshold values
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Iterate through the contours and filter out the ones that might represent railway tracks
    railway_track_contours = []
    for contour in contours:
        # You can add conditions here to filter contours based on area, aspect ratio, etc.
        if cv2.contourArea(contour) > min_area_threshold:
            railway_track_contours.append(contour)

    # Draw the filtered railway track contours on the original image
    result_image = frame.copy()
    cv2.drawContours(result_image, railway_track_contours, -1, (0, 255, 0), 2)  # Green color, line thickness 2

# Show the original image with the detected railway track
    cv2.imshow('Detected Railway Track', result_image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cv2.waitKey(0)
cv2.destroyAllWindows()
# cap.release()