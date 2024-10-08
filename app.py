import streamlit as st
import sqlite3
from datetime import datetime
import cv2

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

def main():
    init_embedding_db()
    init_detection_db()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_facenet_model(device)

    data_path = "image_persons"
    new_persons = check_for_new_persons(data_path)
    if new_persons:
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
