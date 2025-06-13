import cv2
import os
import pandas as pd

# Folder containing the videos
video_folder = r"C:\Users\TRIOS\OneDrive - KTH\Thesis - Giorgio Coraglia\Final Video"

# List to store the results
results = []

# Get all video files in the folder
video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)

    # Loop to allow rewatching the video
    while True:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Video', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        print(f"\nClassify the video: {video_file}")
        print("Press 's' for Sagittal, 'c' for Coronal, 'r' to rewatch the video")

        key = ''
        while key not in ['s', 'c', 'r']:
            key = input("(s/c/r): ").lower()

        if key == 'r':
            continue  # Rewatch the video
        elif key == 's':
            fall_type = 'Sagittal'
        else:
            fall_type = 'Coronal'

        results.append({'Video': video_file, 'Fall Type': fall_type})
        break  # Exit inner loop and move to the next video

# Create a DataFrame and save it as Excel
results_df = pd.DataFrame(results)
results_df.to_excel('fall_classification.xlsx', index=False)

print("\nClassification completed. File saved as 'fall_classification.xlsx'")
