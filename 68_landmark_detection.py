import cv2
import dlib

# Load the facial landmark predictor for face alignment
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You should download this model

# Load the input image
input_image_path = "user.jpeg"  # Replace with the path to your input image
image = cv2.imread(input_image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection using dlib
detector = dlib.get_frontal_face_detector()
faces = detector(gray)

# Iterate over detected faces
for face in faces:
    landmarks = predictor(gray, face)

    # Draw facial landmarks on the image
    for n in range(0, 68):  # Assuming 68 landmarks are being used (adjust if needed)
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red circles at each landmark

# Save or display the image with landmarks
output_image_path = r"C:\Users\arshpreet\OneDrive\Desktop\face_recognization_research_papers\face_recognization_detection\output_image.jpg"  # Replace with the path where you want to save the output image
cv2.imwrite(output_image_path, image)

# Display the image with landmarks (optional)
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
