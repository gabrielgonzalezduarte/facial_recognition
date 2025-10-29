import sqlite3
import os
import cv2
import numpy as np
from datetime import datetime

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('faces.db', check_same_thread=False)
        self.create_table()
    
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_data BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def save_face(self, name, face_data):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO faces (name, face_data) 
            VALUES (?, ?)
        ''', (name, face_data))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_all_faces(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, name, face_data FROM faces')
        return cursor.fetchall()
    
    def delete_face(self, face_id):
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
        self.conn.commit()
    
    def close(self):
        self.conn.close()