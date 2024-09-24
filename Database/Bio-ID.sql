CREATE DATABASE face_detection;
USE face_detection;
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    face_image_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    
    INSERT INTO users (username, face_image_path)
VALUES ('user1', '/E:\Bio-ID-main\Database\face_detection\users\abdullah_alahmariusers/to/user1_face_image.jpg'),
       ('user2', '/E:\Bio-ID-main\Database\face_detection\users\abdullah_alawwad/to/user2_face_image.jpg'),
       ('user3', '/E:\Bio-ID-main\Database\face_detection\users\nawaf_alanezi/to/user3_face_image.jpg');
       SELECT * FROM users;
