

import streamlit as st
import cv2
import face_recognition
import pandas as pd
import numpy as np
import os
from datetime import datetime, date
import pickle
import json
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import time

class StreamlitFaceRecognition:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.attendance_file = "attendance.csv"
        self.encodings_file = "face_encodings.pickle"
        self.config_file = "config.json"
        
        # Load existing data
        self.load_config() # Config pehle load karen
        self.load_encodings()
    
    def load_config(self):
        """Load configuration settings and ensure all keys exist."""
        default_config = {
            "tolerance": 0.6,
            "confidence_threshold": 0.5
        }
        
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                try:
                    self.config = json.load(f)
                except json.JSONDecodeError:
                    self.config = default_config
            
            # Check for missing keys and add them with default values
            updated = False
            for key, value in default_config.items():
                if key not in self.config:
                    self.config[key] = value
                    updated = True
            
            if updated:
                self.save_config()
                
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save configuration settings"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def load_encodings(self):
        """Load pre-computed face encodings"""
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_faces = data['encodings']
                self.known_names = data['names']
    
    def save_encodings(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_faces,
            'names': self.known_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
    
    def add_person_from_image(self, image, person_name):
        """Add a new person from uploaded image"""
        try:
            if person_name in self.known_names:
                return False, f"{person_name} is already registered."

            image_np = np.array(image)
            encodings = face_recognition.face_encodings(image_np)
            
            if encodings:
                self.known_faces.append(encodings[0])
                self.known_names.append(person_name)
                self.save_encodings()
                return True, f"Successfully added {person_name} to the system!"
            else:
                return False, "No face found in the uploaded image!"
                
        except Exception as e:
            return False, f"Error processing image: {str(e)}"
    
    def load_images_from_folder(self):
        """Load all images from the images folder, clearing previous entries"""
        # Clear existing lists to prevent duplicates
        self.known_faces = []
        self.known_names = []
        
        images_folder = "images"
        
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
            return 0, [], "Images folder not found! An 'images' folder has been created for you."
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        loaded_count = 0
        failed_files = []
        
        for filename in os.listdir(images_folder):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                try:
                    file_path = os.path.join(images_folder, filename)
                    image = face_recognition.load_image_file(file_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    person_name = os.path.splitext(filename)[0].replace('_', ' ').title()

                    if encodings and person_name not in self.known_names:
                        self.known_faces.append(encodings[0])
                        self.known_names.append(person_name)
                        loaded_count += 1
                    elif person_name in self.known_names:
                        # Skip if person is already added (e.g., from another photo)
                        continue
                    else:
                        failed_files.append(f"{filename} (no face found)")
                        
                except Exception as e:
                    failed_files.append(f"{filename} (error: {str(e)})")
        
        if loaded_count > 0:
            self.save_encodings()
        
        return loaded_count, failed_files, ""
    
    def recognize_face(self, image):
        """Recognize faces in an image"""
        try:
            image_np = np.array(image)
            face_locations = face_recognition.face_locations(image_np)
            face_encodings = face_recognition.face_encodings(image_np, face_locations)
            
            results = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    self.known_faces, face_encoding, tolerance=self.config['tolerance']
                )
                name = "Unknown"
                confidence = 0
                
                if True in matches:
                    face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (top, right, bottom, left)
                })
            
            return results
            
        except Exception as e:
            st.error(f"An error occurred during face recognition: {e}")
            return []
    
    def mark_attendance(self, name):
        """Mark attendance for a person"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        if os.path.exists(self.attendance_file):
            df = pd.read_csv(self.attendance_file)
        else:
            df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status'])
        
        today_attendance = df[(df['Name'] == name) & (df['Date'] == date_str)]
        
        if today_attendance.empty:
            new_record = pd.DataFrame({
                'Name': [name], 'Date': [date_str], 'Time': [time_str], 'Status': ['Present']
            })
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(self.attendance_file, index=False)
            return True, f"Attendance marked for {name} at {time_str}"
        else:
            return False, f"{name} has already been marked present today."
    
    def get_attendance_data(self):
        """Get attendance data"""
        if os.path.exists(self.attendance_file):
            return pd.read_csv(self.attendance_file)
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="Face Recognition Attendance", page_icon="👤", layout="wide")
    
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #4F8BF9; text-align: center; font-weight: bold; }
    .sub-header { font-size: 1.75rem; color: #1E3A8A; }
    </style>
    """, unsafe_allow_html=True)
    
    system = StreamlitFaceRecognition()
    
    st.markdown('<h1 class="main-header"> Face Recognition Attendance System</h1>', unsafe_allow_html=True)
    
    st.sidebar.title(" Navigation")
    page = st.sidebar.radio("Choose a page:", [" Home", " Add Person", " Recognition", " Attendance Records", " Analytics", "⚙️ Settings"])
    
    if page == " Home":
        st.markdown('<h2 class="sub-header">Dashboard</h2>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        today_df = system.get_attendance_data()
        today_count = 0
        if not today_df.empty:
            today_count = len(today_df[today_df['Date'] == date.today().strftime("%Y-%m-%d")])

        col1.metric(" Registered People", len(system.known_names))
        col2.metric(" Today's Attendance", today_count)
        col3.metric(" Total Records", len(today_df))
        
        st.markdown("---")
        st.markdown('<h3 class="sub-header"> Quick Actions</h3>', unsafe_allow_html=True)
        
        if st.button(" Reload All People from 'images' Folder", type="primary"):
            with st.spinner("Processing... This will remove old entries and load new ones."):
                loaded_count, failed_files, message = system.load_images_from_folder()
                if message:
                    st.info(message)
                if loaded_count > 0:
                    st.success(f" Successfully loaded {loaded_count} unique people!")
                    if failed_files:
                        st.warning(f"⚠️ Failed to process: {', '.join(failed_files)}")
                else:
                    st.error(" No new people were loaded.")
        
        if st.button(" Clear All Data"):
            st.warning("This will delete all registered faces and attendance records. Are you sure?")
            if st.button("⚠️ Yes, I'm sure. Delete everything."):
                if os.path.exists(system.encodings_file): os.remove(system.encodings_file)
                if os.path.exists(system.attendance_file): os.remove(system.attendance_file)
                if os.path.exists(system.config_file): os.remove(system.config_file)
                st.success("All data has been cleared!")
                time.sleep(1)
                st.rerun()

    elif page == " Add Person":
        st.markdown('<h2 class="sub-header">Register New Person</h2>', unsafe_allow_html=True)
        person_name = st.text_input("Enter Person's Name", placeholder="e.g., Ali Khan")
        uploaded_file = st.file_uploader("Upload a clear photo of the person", type=['jpg', 'jpeg', 'png'])

        if st.button(" Add Person", type="primary") and uploaded_file and person_name:
            with st.spinner("Processing image..."):
                image = Image.open(uploaded_file)
                success, message = system.add_person_from_image(image, person_name)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        elif st.button(" Add Person", type="primary"):
            st.warning("Please provide both name and image.")

    elif page == " Recognition":
        st.markdown('<h2 class="sub-header">Mark Attendance</h2>', unsafe_allow_html=True)
        if not system.known_names:
            st.warning("⚠️ No people registered! Please add people from the 'Add Person' page first.")
        else:
            uploaded_file = st.file_uploader("Upload an image for recognition", type=['jpg', 'jpeg', 'png', 'bmp'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Image for Recognition", width=400)
                
                if st.button(" Recognize & Mark Attendance", type="primary"):
                    with st.spinner("Recognizing faces..."):
                        results = system.recognize_face(image)
                    
                    if results:
                        st.success(f"Found {len(results)} face(s)!")
                        for i, result in enumerate(results):
                            name = result['name']
                            confidence = result['confidence']
                            
                            if name != "Unknown" and confidence >= system.config['confidence_threshold']:
                                st.write(f" **{name}** (Confidence: {confidence:.2f})")
                                success, message = system.mark_attendance(name)
                                if success:
                                    st.success(message)
                                else:
                                    st.info(message)
                            else:
                                st.warning(f" **Unknown Person** or low confidence (Confidence: {confidence:.2f})")
                    else:
                        st.warning("No faces detected in the image.")

    elif page == " Attendance Records":
        st.markdown('<h2 class="sub-header">Attendance Log</h2>', unsafe_allow_html=True)
        df = system.get_attendance_data()
        if df.empty:
            st.info(" No attendance records found!")
        else:
            col1, col2 = st.columns(2)
            date_filter = col1.date_input(" Filter by Date", value=None)
            name_filter = col2.selectbox(" Filter by Person", options=["All"] + sorted(list(df['Name'].unique())))
            
            filtered_df = df.copy()
            if date_filter:
                filtered_df = filtered_df[filtered_df['Date'] == date_filter.strftime("%Y-%m-%d")]
            if name_filter != "All":
                filtered_df = filtered_df[filtered_df['Name'] == name_filter]
            
            st.dataframe(filtered_df.sort_values(by=['Date', 'Time'], ascending=False), use_container_width=True)
            if not filtered_df.empty:
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(" Download Records", csv, "attendance.csv", "text/csv")

    elif page == " Analytics":
        st.markdown('<h2 class="sub-header">Attendance Analytics</h2>', unsafe_allow_html=True)
        df = system.get_attendance_data()
        if df.empty:
            st.info(" No data available for analytics!")
        else:
            st.markdown("### Daily Attendance Trend")
            daily_counts = df.groupby('Date').size().reset_index(name='Count')
            fig_date = px.line(daily_counts, x='Date', y='Count', title=" Daily Attendance", markers=True)
            st.plotly_chart(fig_date, use_container_width=True)

            st.markdown("### Attendance by Person")
            person_counts = df.groupby('Name').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
            fig_person = px.bar(person_counts, x='Name', y='Count', title=" Total Attendance per Person")
            st.plotly_chart(fig_person, use_container_width=True)

    elif page == "⚙️ Settings":
        st.markdown('<h2 class="sub-header">System Settings</h2>', unsafe_allow_html=True)
        st.markdown("###  Recognition Settings")
        tolerance = st.slider("Face Recognition Tolerance", 0.1, 1.0, system.config['tolerance'], 0.05, help="Lower = More strict, Higher = More lenient")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, system.config.get('confidence_threshold', 0.5), 0.05, help="Minimum confidence required to mark attendance.")
        
        if st.button(" Save Settings"):
            system.config['tolerance'] = tolerance
            system.config['confidence_threshold'] = confidence_threshold
            system.save_config()
            st.success("Settings saved successfully!")

        st.markdown("---")
        st.markdown("###  Registered People")
        if system.known_names:
            for name in sorted(system.known_names):
                st.write(f"- {name}")
        else:
            st.info("No one is registered yet.")

if __name__ == "__main__":
    main()