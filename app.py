import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import sqlite3
from datetime import datetime
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1  
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  


def load_facenet_model(device):
    model = InceptionResnetV1(pretrained='vggface2').eval()  
    model.to(device)
    return model


def preprocess_image(image, face_cascade_path='haarcascade_frontalface_default.xml'):
    
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

    
    faces = face_cascade.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    
    x, y, w, h = faces[0]
    face = img_rgb[y:y+h, x:x+w]

    
    face_pil = Image.fromarray(face)

    
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    
    face_tensor = transform(face_pil)
    face_tensor = face_tensor.unsqueeze(0)  

    return face_tensor


def init_embedding_db():
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            person_name TEXT PRIMARY KEY,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()


def save_embedding_to_db(person_name, embedding):
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO embeddings (person_name, embedding) VALUES (?, ?)',
              (person_name, embedding.tobytes()))
    conn.commit()
    conn.close()


def load_embeddings_from_db():
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute('SELECT person_name, embedding FROM embeddings')
    rows = c.fetchall()
    conn.close()

    known_face_embeddings = []
    class_names = []

    for row in rows:
        person_name = row[0]
        embedding = np.frombuffer(row[1], dtype=np.float32)
        known_face_embeddings.append(embedding)
        class_names.append(person_name)

    known_face_embeddings = np.array(known_face_embeddings)
    return known_face_embeddings, class_names


def check_for_new_persons(data_path):
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()

    
    c.execute('SELECT person_name FROM embeddings')
    existing_persons = set([row[0] for row in c.fetchall()])

    
    data_persons = set(os.listdir(data_path))

    
    new_persons = data_persons - existing_persons
    conn.close()

    return list(new_persons)


def generate_and_store_new_embeddings(model, new_persons, data_path, device):
    for person_name in tqdm(new_persons, desc="Generating embeddings for new persons"):
        person_dir = os.path.join(data_path, person_name)
        if os.path.isdir(person_dir):
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                try:
                    image = Image.open(image_path)
                    face_tensor = preprocess_image(image)  
                    face_tensor = face_tensor.to(device)

                    
                    with torch.no_grad():
                        embedding = model(face_tensor).cpu().numpy()

                    
                    save_embedding_to_db(person_name, embedding)

                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")


def recognize_face(model, face_tensor, known_face_embeddings, class_names, device, threshold=0.6):
    
    face_tensor = face_tensor.to(device)

    
    with torch.no_grad():
        embedding = model(face_tensor).cpu().numpy()

    
    similarities = cosine_similarity(embedding, known_face_embeddings)
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[0][best_match_idx]

    if best_match_score >= threshold:
        return class_names[best_match_idx]  
    else:
        return "Unknown"  


def init_detection_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT,
            datetime TEXT
        )
    ''')
    conn.commit()
    conn.close()


def log_detection(person_name):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO detections (person_name, datetime) VALUES (?, ?)', (person_name, timestamp))
    conn.commit()
    conn.close()


detecting = False

def main():
    

    
    init_embedding_db()
    init_detection_db()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model = load_facenet_model(device)

    
    data_path = "image_persons"

    
    new_persons = check_for_new_persons(data_path)
    if new_persons:
        print("Found New Persons Putting them into db.")
        generate_and_store_new_embeddings(model, new_persons, data_path, device)

    
    
    known_face_embeddings, class_names = load_embeddings_from_db()
    
    
    st.markdown("<h1 style='text-align: center; color: grey;'>Person Recognition</h1>", unsafe_allow_html=True)
    
    
    FRAME_WINDOW = st.empty()  
    prediction_placeholder = st.empty()  
    button_placeholder = st.empty()  

    
    bg_image = cv2.imread(r"bg-image.webp")
    bg_image = cv2.resize(bg_image, (640, 480))
    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(bg_image)

    global detecting  

    
    with button_placeholder:
        st.write("")  
        cols = st.columns(5)  

        
        if cols[1].button('Start Detection'):
            if 'detecting' not in st.session_state or not st.session_state.detecting:
                st.session_state.detecting = True  

        
        if cols[3].button('Stop Detection'):
            if 'detecting' in st.session_state and st.session_state.detecting:
                st.session_state.detecting = False  

    if 'detecting' in st.session_state and st.session_state.detecting:
        
        cap = cv2.VideoCapture(0)

        while cap.isOpened() and st.session_state.detecting:  
            ret, frame = cap.read()
            if not ret:
                st.write("Unable to read camera feed.")
                break
            
            try:
                face_tensor = preprocess_image(frame)
                predicted_class = recognize_face(model, face_tensor, known_face_embeddings, class_names, device)
                log_detection(predicted_class)
            except ValueError:
                predicted_class = "No face detected"
            
            
            if predicted_class != "Unknown":
                text = f"Prediction: Welcome {predicted_class}" if predicted_class != "No face detected" else predicted_class
            else:
                text = "Prediction: Unknown Person"
            
            prediction_placeholder.markdown(f"<p style='position:fixed; bottom:10px; left:10px; font-size:20px;'>{text}</p>", unsafe_allow_html=True)

            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)  

        cap.release()  

if __name__ == "__main__":
    main()
