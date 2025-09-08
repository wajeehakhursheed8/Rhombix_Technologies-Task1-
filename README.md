Face Recognition Attendance System
A web-based face recognition attendance system built with Streamlit, utilizing face_recognition for face deteyr ction and recognition, and pandas for attendance tracking. Developed during an internship at RomVix Technologies, this application allows users to register people, mark attendance using facial recognition, and view attendance analytics.
Features

Register People: Add individuals by uploading images or loading from an images folder.
Face Recognition: Detect and recognize faces in images to mark attendance.
Attendance Tracking: Log attendance with date, time, and status, stored in attendance.csv.
Analytics: Visualize daily attendance trends and per-person attendance using Plotly charts.
Settings: Adjust face recognition tolerance and confidence threshold.

Prerequisites

Python 3.8+
Required libraries: streamlit, face_recognition, opencv-python, pandas, numpy, pillow, plotly, pickle
A webcam or image files for face recognition


Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Run the application:
streamlit run main.py



File Structure

main.py: Main application script containing the Streamlit app and face recognition logic.
images/: Folder to store images for registering people (auto-created if not present).
attendance.csv: Stores attendance records (auto-generated).
face_encodings.pickle: Stores pre-computed face encodings (auto-generated).
config.json: Stores configuration settings like tolerance and confidence threshold (auto-generated).
requirements.txt: Lists all required Python libraries.

Usage

Home: View dashboard metrics and reload people from the images folder or clear all data.
Add Person: Upload an image and enter a name to register a new person.
Recognition: Upload an image to detect faces and mark attendance.
Attendance Records: Filter and download attendance logs by date or person.
Analytics: View daily attendance trends and per-person attendance charts.
Settings: Adjust recognition tolerance and confidence threshold, and view registered people.

Notes

Place images in the images folder with filenames as names (e.g., Ali_Khan.jpg) for batch loading.
Ensure images are clear and contain a single face for best recognition results.
The system skips duplicate names during batch loading to avoid conflicts.
Attendance is marked only once per day per person.

Dependencies
Listed in requirements.txt:
streamlit==1.25.0
face_recognition==1.3.0
opencv-python==4.8.0
pandas==2.0.3
numpy==1.24.3
pillow==10.0.0
plotly==5.15.0


