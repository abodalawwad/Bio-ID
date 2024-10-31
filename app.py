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
import base64
import requests
import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

# i18n dictionary for translations
i18n = {
    'en': {
        'login': '🔐 Login',
        'username': 'Username',
        'password': 'Password',
        'login_button': '➡️ Login',
        'login_successful': '✅ Login successful!',
        'login_error': '❌ Invalid username or password',
        'face_detection_page': '👁️ Face Verification',
        'face_detection_success': '✅ Face verified successfully!',
        'no_face_detection': '⚠️ No face detected',
        'failed_read_from_camera': '❌ Failed to read from camera',
        'greeting': '👋 Welcome',
        'book_search_section': '🔍 Book Search',
        'enter_book_title': '📚 Enter a book title:',
        'author': '✍️ Author',
        'no_books_found': '📭 No books found.',
        'logout': '🚪 Logout',
        'logout_successful': '👋 Logout successful'
    },
    'ar': {
        'login': '🔐 تسجيل الدخول',
        'username': 'اسم المستخدم',
        'password': 'كلمة المرور',
        'login_button': '➡️ دخول',
        'login_successful': '✅ تم تسجيل الدخول بنجاح!',
        'login_error': '❌ اسم المستخدم أو كلمة المرور غير صحيحة',
        'face_detection_page': '👁️ التحقق من الوجه',
        'face_detection_success': '✅ تم التحقق من الوجه بنجاح!',
        'no_face_detection': '⚠️ لم يتم اكتشاف وجه',
        'failed_read_from_camera': '❌ فشل في القراءة من الكاميرا',
        'greeting': '👋 مرحبا',
        'book_search_section': '🔍 بحث الكتب',
        'enter_book_title': '📚 أدخل عنوان الكتاب:',
        'author': '✍️ المؤلف',
        'no_books_found': '📭 لم يتم العثور على كتب.',
        'logout': '🚪 تسجيل الخروج',
        'logout_successful': '👋 تم تسجيل الخروج بنجاح'
    }
}

# Utility functions for face recognition
def load_facenet_model(device):
    """Load and initialize the FaceNet model"""
    model = InceptionResnetV1(pretrained='vggface2').eval()  
    model.to(device)
    return model

def preprocess_image(image, face_cascade_path='haarcascade_frontalface_default.xml'):
    """Preprocess image for face detection and recognition"""
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

def recognize_face(model, face_tensor, known_face_embeddings, class_names, device, threshold=0.6):
    """Recognize a face by comparing it with known face embeddings"""
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        embedding = model(face_tensor).cpu().numpy()
    
    if len(known_face_embeddings) == 0:
        return "Unknown"
        
    similarities = cosine_similarity(embedding, known_face_embeddings)
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[0][best_match_idx]
    
    if best_match_score >= threshold:
        return class_names[best_match_idx]
    else:
        return "Unknown"

# Database logging utility
def log_detection(person_name):
    try:
        with sqlite3.connect('detections.db', timeout=10) as conn:
            c = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute('INSERT INTO detections (person_name, datetime) VALUES (?, ?)', 
                     (person_name, timestamp))
            conn.commit()
    except Exception as e:
        # Use st.error only once and place it in a visible location
        if 'error_shown' not in st.session_state:
            st.error(f"Error logging detection: {e}")
            st.session_state.error_shown = True

# OpenAI utility
def generate_text(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"An error occurred: {e}"

st.set_page_config(layout="centered", page_title="Book Explorer", page_icon="📚")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def get_book_info(title):
    url = f'https://openlibrary.org/search.json?title={title}'
    response = requests.get(url)
    return response.json()

def apply_language_styles():
    if st.session_state.lang == 'ar':
        st.markdown("""
            <style>
            .stApp {
                direction: rtl;
                text-align: right;
            }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                direction: ltr;
                text-align: left;
            }
            </style>
            """, unsafe_allow_html=True)

def add_header():
    # Get the current language from session state
    current_lang = st.session_state.get('lang', 'en')  # Default to 'en' if not set
    
    # Language label translations
    lang_label = {
        'en': 'Language',
        'ar': 'اللغة'
    }

    # Define header text for both languages
    header_text = {
        'en': {
            'title': '📚 Book Explorer',
            'subtitle': '🔍 Discover, Search, and Explore Books Instantly'
        },
        'ar': {
            'title': '📚 مستكشف الكتب',
            'subtitle': '🔍 اكتشف وابحث واستكشف الكتب فوراً'
        }
    }

    # Set text direction based on language
    text_direction = 'rtl' if current_lang == 'ar' else 'ltr'

    # Use a default language if the current language is not found
    if current_lang not in header_text:
        current_lang = 'en'

    st.markdown(f"""
        <style>
        .header {{
            background: linear-gradient(to right, #4A00E0, #8E2DE2, #00C9FF);
            padding: 20px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-radius: 0 0 8px 8px;
            margin-bottom: 20px;
            direction: {text_direction};
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 999;
        }}
        .stApp > header + div {{
            padding-top: 130px;
        }}
        .logo {{
            width: 60px;
            height: 60px;
        }}
        .title-container {{
            color: white;
            flex-grow: 1;
            text-align: center;
        }}
        .main-title {{
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .subtitle {{
            font-size: 16px;
            opacity: 0.9;
        }}
        .lang-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-right: 20px;
        }}
        .lang-label {{
            color: white;
            font-size: 12px;
            margin-bottom: 5px;
        }}
        .lang-buttons {{
            display: flex;
            gap: 5px;
        }}
        .lang-button {{
            padding: 5px 10px;
            border: 1px solid white;
            border-radius: 4px;
            color: white;
            background: transparent;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s ease;
        }}
        .lang-button:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}
        .lang-button.active {{
            background: white;
            color: #4A00E0;
        }}
        </style>
        <div class="header">
            <img src="https://cdn-icons-png.flaticon.com/512/2232/2232688.png" class="logo" alt="Book Explorer Logo">
            <div class="title-container">
                <div class="main-title">{header_text[current_lang]['title']}</div>
                <div class="subtitle">{header_text[current_lang]['subtitle']}</div>
            </div>
            <div class="lang-container">
                <div class="lang-label">{lang_label[current_lang]}</div>
                <div class="lang-buttons">
                    <a 
                        href="?lang=en" 
                        class="lang-button {'active' if current_lang == 'en' else ''}"
                        style="text-decoration: none;"
                    >EN</a>
                    <a 
                        href="?lang=ar" 
                        class="lang-button {'active' if current_lang == 'ar' else ''}"
                        style="text-decoration: none;"
                    >عربي</a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def add_footer():
    footer_style = """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            padding: 10px 0;
            text-align: center;
            z-index: 999;
        }
        .footer-content {
            max-width: 800px;
            margin: 0 auto;
            padding: 0 20px;
        }
        .github-link {
            color: #333;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        .github-link:hover {
            color: #4A00E0;
        }
        .github-icon {
            vertical-align: middle;
            margin-right: 5px;
        }
        /* Add padding to main content to prevent footer overlap */
        [data-testid="stAppViewContainer"] {
            padding-bottom: 60px;
        }
    </style>
    """

    footer_html = f"""
    <div class="footer">
        <div class="footer-content">
            © 2024 Book Explorer | 
            <a href="https://github.com/abodalawwad/Bio-ID" target="_blank" class="github-link">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" 
                     width="30" 
                     height="30" 
                     class="github-icon">
                GitHub
            </a>
        </div>
    </div>
    """

    st.markdown(footer_style + footer_html, unsafe_allow_html=True)

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

def is_valid_user(username, password):
    # First check database
    with sqlite3.connect('users.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE LOWER(username) = LOWER(?) AND password = ?', (username, password))
        user = c.fetchone()
        if user is not None:
            return True
    
    # Then check hardcoded users
    valid_users = {
        'Alahmari': '1234',
        'user1': 'mypassword'
    }
    return valid_users.get(username) == password

def login_page():
    apply_language_styles()
    add_header()
    
    # Add spacing after header
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # Get current language with fallback to English
    current_lang = st.session_state.get('lang', 'en')
    if current_lang not in ['en', 'ar']:
        current_lang = 'en'
        st.session_state.lang = 'en'
    
    # Create three columns for centering
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <style>
            /* Animated Background */
            .stApp {
                background: linear-gradient(-45deg, #0F172A, #1E293B, #312E81, #1E3A8A);
                background-size: 400% 400%;
                animation: gradientBG 15s ease infinite;
                direction: """ + ('rtl' if current_lang == 'ar' else 'ltr') + """;
                text-align: """ + ('right' if current_lang == 'ar' else 'left') + """;
            }
            
            @keyframes gradientBG {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            /* Login Container with Glass Effect */
            [data-testid="column"] > div {
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                padding: 3rem;
                backdrop-filter: blur(12px);
                box-shadow: 0 0 45px rgba(0, 0, 0, 0.3);
                animation: floatIn 0.4s ease-out;
                width: 100%;
                max-width: 500px;
                margin: 0 auto;
            }
            
            @keyframes floatIn {
                0% {
                    opacity: 0;
                    transform: translateY(20px) scale(0.95);
                }
                100% {
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }
            }
            
            /* Title with Gradient */
            h1 {
                background: linear-gradient(120deg, #fff, #94a3b8, #fff);
                background-size: 200% auto;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: shine 5s linear infinite;
                font-size: 2.5rem !important;
                margin-bottom: 2rem !important;
                text-align: center;
            }
            
            /* Input Container */
            .stTextInput {
                margin-bottom: 1.5rem;
                position: relative;
                animation: slideIn 0.6s ease-out backwards;
            }
            
            /* Updated Input Field Styles */
            .stTextInput > div > div > input {
                width: 160% !important;
                padding: 1.0rem 0.5rem !important;
                height: 0rem !important;
                background: rgba(255, 255, 255, 0.05) !important;
                border: none !important;
                border-radius: 0px !important;
                color: white !important;
                font-size: 1.2rem !important;
                transition: all 0.9s ease !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                margin-bottom: 1.5rem !important;
            }
            
            /* Input Focus Animation */
            .stTextInput > div > div > input:focus {
                background: rgba(255, 255, 255, 0.08) !important;
                border-color: #6366f1 !important;
                box-shadow: 0 0 15px rgba(99, 102, 241, 0.3) !important;
            }
            
            /* Updated Button Styles */
            .stButton > button {
                width: 190% !important;
                height: 4rem !important;
                background: linear-gradient(45deg, #6366f1, #4f46e5) !important;
                color: white !important;
                border: none !important;
                border-radius: 2px !important;
                font-size: 0.9rem !important;
                font-weight: 500 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.1em !important;
                transition: all 0.3s ease !important;
                margin-top: 1rem !important;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
            }
            
            /* Alert Messages */
            .stAlert {
                background: rgba(255, 255, 255, 0.05) !important;
                backdrop-filter: blur(10px) !important;
                border-radius: 8px !important;
                animation: slideUp 0.5s ease-out;
            }
            
            @keyframes slideUp {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Login form
        st.title(i18n[current_lang]['login'])
        
        with st.form("login_form"):
            username = st.text_input(
                i18n[current_lang]['username'],
                key="username_input",
                placeholder=i18n[current_lang]['username']
            )
            
            password = st.text_input(
                i18n[current_lang]['password'],
                type="password",
                key="password_input",
                placeholder=i18n[current_lang]['password']
            )
            
            submit_button = st.form_submit_button(i18n[current_lang]['login_button'])
            
            if submit_button:
                if is_valid_user(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success(i18n[current_lang]['login_successful'])
                    st.rerun()
                else:
                    st.error(i18n[current_lang]['login_error'])
    
    add_footer()

def welcome_page():
    apply_language_styles()
    add_header()
    
    st.title(f"{i18n.get(st.session_state.lang).get('greeting')} {st.session_state['username']}!")
    st.subheader(i18n.get(st.session_state.lang).get('book_search_section', '🔍 Book Search'))
    
    # Book Search Section
    book_title = st.text_input(i18n.get(st.session_state.lang).get('enter_book_title', 'Enter a book title:'))
    if book_title:
        book_data = get_book_info(book_title)
        if book_data['docs']:
            book = book_data['docs'][0]
            st.subheader(book.get('title', 'No title available'))
            st.write(f"{i18n.get(st.session_state.lang).get('author', 'Author')}: {', '.join(book.get('author_name', ['Unknown']))}")
            cover_id = book.get('cover_i')
            if cover_id:
                st.image(f"http://covers.openlibrary.org/b/id/{cover_id}-L.jpg", caption=book['title'])
        else:
            st.error(i18n.get(st.session_state.lang).get('no_books_found', 'No books found.'))
    
    # Logout button
    if st.button(i18n.get(st.session_state.lang).get('logout')):
        st.success(i18n.get(st.session_state.lang).get('logout_successful'))
        st.session_state.clear()
        st.rerun()
    
    add_footer()

def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #2c3e50;
            background-image: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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

def init_detection_db():
    try:
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
    except Exception as e:
        print(f"Error initializing detection database: {e}")

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

def save_embedding_to_db(person_name, embedding):
    with sqlite3.connect('embeddings.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO embeddings (person_name, embedding) VALUES (?, ?)',
                  (person_name, embedding.tobytes()))
        conn.commit()

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

def face_detection_page():
    apply_language_styles()
    add_header()
    st.title(i18n.get(st.session_state.lang).get('face_detection_page', 'Face Verification'))
    
    FRAME_WINDOW = st.image([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_facenet_model(device)
    known_face_embeddings, class_names = load_embeddings_from_db()
    
    # Add a placeholder for status messages
    status_placeholder = st.empty()
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera. Please check your camera connection.")
            return
            
        no_face_detected = False
        warning_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error(i18n.get(st.session_state.lang).get('failed_read_from_camera', 'Failed to read from camera'))
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            
            try:
                face_tensor = preprocess_image(frame)
                recognized_person = recognize_face(model, face_tensor, known_face_embeddings, class_names, device)
                
                if recognized_person == st.session_state["username"]:
                    log_detection(recognized_person)
                    status_placeholder.success(i18n.get(st.session_state.lang).get('face_detection_success', 'Face verified successfully!'))
                    st.session_state["face_recognized"] = True
                    cap.release()
                    st.rerun()
                    break
                else:
                    if no_face_detected:
                        warning_placeholder.empty()
                    no_face_detected = False
                    status_placeholder.warning("Face not recognized. Please try again.")
                    
            except ValueError:
                if not no_face_detected:
                    warning_placeholder.warning(i18n.get(st.session_state.lang).get('no_face_detection', 'No face detected'))
                    no_face_detected = True
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
    
    add_footer()

    # Add a cancel button to skip face verification (for testing purposes)
    if st.button("⏭️ Skip Face Verification (Debug)"):
        st.session_state["face_recognized"] = True
        st.rerun()

def set_animated_theme():
    st.markdown("""
        <style>
        /* Base Styles and Animations */
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes slideInFromTop {
            0% {
                transform: translateY(-20px);
                opacity: 0;
            }
            100% {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        
        /* Main App Container */
        .stApp {
            background: linear-gradient(-45deg, #0F172A, #1E293B, #1E3A8A, #312E81);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }
        
        /* Header Animation */
        .stHeader {
            animation: slideInFromTop 0.5s ease-out;
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(10px);
        }
        
        /* Sidebar Animation */
        [data-testid="stSidebar"] {
            animation: slideInFromTop 0.5s ease-out;
            background: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(10px);
        }
        
        /* Card Container Animation */
        .element-container {
            animation: fadeIn 0.6s ease-out;
            transition: transform 0.3s ease;
        }
        
        .element-container:hover {
            transform: translateY(-2px);
        }
        
        /* Button Animations */
        .stButton > button {
            background: linear-gradient(45deg, #3B82F6, #1D4ED8);
            border: none;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                120deg,
                transparent,
                rgba(255, 255, 255, 0.2),
                transparent
            );
            animation: shimmer 2s infinite;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
            animation: pulse 1s infinite;
        }
        
        /* Input Field Animations */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background: rgba(30, 41, 59, 0.5);
            border: 2px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            color: white;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #3B82F6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
            transform: translateY(-1px);
        }
        
        /* Select Box Animation */
        .stSelectbox > div > div {
            transition: all 0.3s ease;
        }
        
 .stSelectbox > div > div:hover {
            border-color: #3B82F6;
            transform: translateY(-1px);
        }
        
        /* Progress Bar Animation */
        .stProgress > div > div > div > div {
            background: linear-gradient(-45deg, #3B82F6, #1D4ED8);
            background-size: 200% 200%;
            animation: gradient 2s ease infinite;
        }
        
        /* Alert Animation */
        .stAlert {
            animation: slideInFromTop 0.5s ease-out;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(10px);
        }
        
        /* Success Message Animation */
        .element-container.css-1e5imcs.e1tzin5v1 {
            animation: fadeIn 0.5s ease-out;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 8px;
        }
        
        /* Error Message Animation */
        .element-container.css-1offfwp.e1tzin5v1 {
            animation: fadeIn 0.5s ease-out;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            border-radius: 8px;
        }
        
        /* Metric Value Animation */
        [data-testid="stMetricValue"] {
            animation: fadeIn 0.8s ease-out;
        }
        
        /* Table Animation */
        .stTable {
            animation: fadeIn 0.8s ease-out;
            background: rgba(30, 41, 59, 0.3);
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Dataframe Animation */
        .dataframe {
            animation: fadeIn 0.8s ease-out;
            transition: all 0.3s ease;
        }
        
        .dataframe:hover {
            transform: scale(1.01);
        }
        
        /* File Uploader Animation */
        .stFileUploader {
            animation: fadeIn 0.8s ease-out;
            border: 2px dashed rgba(148, 163, 184, 0.2);
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        .stFileUploader:hover {
            border-color: #3B82F6;
            transform: translateY(-2px);
        }
        
        /* Checkbox Animation */
        .stCheckbox > label > div[role="checkbox"] {
            transition: all 0.3s ease;
        }
        
        .stCheckbox > label > div[role="checkbox"]:hover {
            transform: scale(1.1);
        }
        
        /* Radio Button Animation */
        .stRadio > div > label > div:first-child {
            transition: all 0.3s ease;
        }
        
        .stRadio > div > label > div:first-child:hover {
            transform: scale(1.1);
        }
        
        /* Slider Animation */
        .stSlider > div > div > div > div {
            transition: all 0.3s ease;
        }
        
        .stSlider > div > div > div > div:hover {
            transform: scale(1.2);
        }
        
        /* Header Text Animation */
        h1, h2, h3 {
            background: linear-gradient(120deg, #3B82F6, #1D4ED8, #3B82F6);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient 3s linear infinite;
        }
        
        /* Loading Spinner Animation */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .stSpinner > div {
            animation: spin 1s linear infinite;
            border-color: #3B82F6 transparent transparent transparent;
        }
        
        /* Scrollbar Animation */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(30, 41, 59, 0.3);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0. 3);
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(148, 163, 184, 0.5);
        }
        
        /* Tooltip Animation */
        [data-tooltip]:hover::before {
            animation: fadeIn 0.3s ease-out;
        }
        </style>
    """, unsafe_allow_html=True)

def get_text(key):
    lang = st.session_state.get('lang', 'en')
    return i18n[lang].get(key, i18n['en'][key])  # Fallback to English if translation is missing

def main():
    # Initialize databases at startup
    init_user_db()
    init_embedding_db()
    init_detection_db()
    
    # Reset error flag at the start of each session
    if 'error_shown' not in st.session_state:
        st.session_state.error_shown = False
    
    # Initialize session state
    if 'lang' not in st.session_state:
        st.session_state.lang = 'en'
    
    # Get language from URL parameter and validate it
    query_params = st.query_params
    if 'lang' in query_params:
        new_lang = query_params['lang']
        if new_lang in ['en', 'ar']:  # Only accept valid language codes
            st.session_state.lang = new_lang
    
    # Main application flow
    if not st.session_state.get("logged_in", False):
        login_page()
    elif not st.session_state.get("face_recognized", False):
        face_detection_page()
    else:
        welcome_page()
    
    # Check for new face embeddings
    data_path = "image_persons"
    new_persons = check_for_new_persons(data_path)
    if new_persons:
        print("Found New Persons. Putting them into db.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_facenet_model(device)
        generate_and_store_new_embeddings(model, new_persons, data_path, device)

if __name__ == "__main__":
    main()