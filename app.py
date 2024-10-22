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

if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

i18n = {
  'en': { 
      'login': 'Login',
      'username': 'Username',
      'password': 'Password',
      'successful_login_now_face_detection': 'Login successful! Redirecting to face detection...',
      'login_error': "Invalid username or password",
      'face_detection_page': 'Face Detection',
      'failed_read_from_camera': 'Failed to read from camera',
      'face_detection_success': 'Face recognized! Redirecting to welcome page...',
      'no_face_detection': 'No face detected',
      'logout': 'Logout',
      'logout_sucessful': 'You have been logged out.',
      'greeting': 'Welcome back'
  },
  'ar': {
      'login': 'تسجيل الدخول',
      'username': 'اسم المستخدم',
      'password': 'كلمة المرور',
      'successful_login_now_face_detection': 'تسجيل دخول ناجح, جاري التوجه إلى صفحة التحقق من بصمة الوجه...',
      'login_error': "اسم المستخدم\الرقم السري غير صحيح",
      'face_detection_page': 'التحقق من بصمة الوجه',
      'failed_read_from_camera': 'فشل في الاتصال بالكاميرا',
      'face_detection_success': 'تم التعرف على بصمة الوجه. التوجه إلى صفحة الترحيب',
      'no_face_detection': 'لم يتم الكشف عن وجه للمطابقة',
      'logout': 'تسجيل الخروج',
      'logout_sucessful': 'تم تسجيل الخروج بنجاح',
      'greeting': 'أهلاً بك'

  }
}

def init_user_db():
    with sqlite3.connect('users.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT
            )
        ''')
        conn.commit()

def add_user(username, password):
    with sqlite3.connect('users.db', timeout=10) as conn:
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            pass

def fetch_users():
    with sqlite3.connect('users.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users')
        users = c.fetchall()
    return users

def is_valid_user(username, password):
    with sqlite3.connect('users.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE LOWER(username) = LOWER(?) AND password = ?', (username, password))
        user = c.fetchone()
    return user is not None

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
    with sqlite3.connect('embeddings.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                person_name TEXT PRIMARY KEY,
                embedding BLOB
            )
        ''')
        conn.commit()

def save_embedding_to_db(person_name, embedding):
    with sqlite3.connect('embeddings.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO embeddings (person_name, embedding) VALUES (?, ?)',
                  (person_name, embedding.tobytes()))
        conn.commit()

def load_embeddings_from_db():
    with sqlite3.connect('embeddings.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute('SELECT person_name, embedding FROM embeddings')
        rows = c.fetchall()
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
    with sqlite3.connect('embeddings.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute('SELECT person_name FROM embeddings')
        existing_persons = set([row[0] for row in c.fetchall()])
    data_persons = set(os.listdir(data_path))
    new_persons = data_persons - existing_persons
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
    with sqlite3.connect('detections.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT,
                datetime TEXT
            )
        ''')
        conn.commit()

def log_detection(person_name):
    with sqlite3.connect('detections.db', timeout=10) as conn:
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute('INSERT INTO detections (person_name, datetime) VALUES (?, ?)', (person_name, timestamp))
        conn.commit()

def login_page():
    st.title(i18n.get(st.session_state.lang).get('login'))
    username = st.text_input(i18n.get(st.session_state.lang).get('username'))
    password = st.text_input(i18n.get(st.session_state.lang).get('password'), type="password")
    if st.button(i18n.get(st.session_state.lang).get('login')):
        if is_valid_user(username, password):
            st.session_state["username"] = username
            st.session_state["logged_in"] = True
            st.success(i18n.get(st.session_state.lang).get('successful_login_now_face_detection'))
            st.rerun()
        else:
            st.error(i18n.get(st.session_state.lang).get('login_error'))

    if st.session_state.lang == 'en':
        
        arabic_link_html = """
                           <div style="text-align: center;"> <a href="#" onclick="window.streamlitRPC.setComponentValue({id: 'change_language'})"> العربية </a></div>
         """
        #st.markdown(arabic_link_html , unsafe_allow_html=True)

        if st.button("تغيير إلى العربية"):
            st.session_state.lang = 'ar'
            st.rerun()

    elif st.session_state.lang == 'ar':
        english_link_html = """
                           <div style="text-align: center;"> <a href="#" onclick="window.streamlitRPC.setComponentValue({id: 'change_language'})"> english </a></div>
         """
        #st.markdown(english_link_html , unsafe_allow_html=True)

        if st.button("Change to English"):
            st.session_state.lang = 'en'
            st.rerun()



def face_detection_page():
    st.title(i18n.get(st.session_state.lang).get('face_detection_page'))
    FRAME_WINDOW = st.image([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_facenet_model(device)
    known_face_embeddings, class_names = load_embeddings_from_db()
    cap = cv2.VideoCapture(0)
    no_face_detected = False
    warning_placeholder = st.empty()  # Create a placeholder for the warning message
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error(i18n.get(st.session_state.lang).get('failed_read_from_camera'))
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        try:
            face_tensor = preprocess_image(frame)
            recognized_person = recognize_face(model, face_tensor, known_face_embeddings, class_names, device)
            if recognized_person == st.session_state["username"]:
                log_detection(recognized_person)
                st.success(i18n.get(st.session_state.lang).get('face_detection_success'))
                st.session_state["face_recognized"] = True
                st.rerun()
                break
            else:
                if no_face_detected:
                    warning_placeholder.empty()  # Clear the warning if a face is detected
                no_face_detected = False
        except ValueError:
            if not no_face_detected:
                warning_placeholder.warning(i18n.get(st.session_state.lang).get('no_face_detection'))
                no_face_detected = True
    cap.release()
def welcome_page():
    st.title(f"{i18n.get(st.session_state.lang).get('greeting')} {st.session_state['username']}!")
    if st.button(i18n.get(st.session_state.lang).get('logout')):
        st.success(i18n.get(st.session_state.lang).get('logout_sucessful'))
        st.session_state.clear()
        st.rerun()

def main():
    init_user_db()
    init_embedding_db()
    init_detection_db()
    users_to_add = [
        ("Alahmari", "1234"),
        ("Alawad", "1234"),
        ("Alenazi", "1234")
    ]
    for username, password in users_to_add:
        add_user(username, password)
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "face_recognized" not in st.session_state:
        st.session_state["face_recognized"] = False
    if not st.session_state["logged_in"]:
        login_page()
    elif not st.session_state["face_recognized"]:
        face_detection_page()
    else:
        welcome_page()



    data_path = "image_persons"
    new_persons = check_for_new_persons(data_path)
    if new_persons:
        print("Found New Persons. Putting them into db.")
        model = load_facenet_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        generate_and_store_new_embeddings(model, new_persons, data_path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

if __name__ == "__main__":
    main()
