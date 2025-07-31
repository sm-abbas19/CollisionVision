import os
from urllib.parse import quote_plus

class Config:
    # Update these credentials based on your PostgreSQL and DataGrip settings
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'collision_detection_db')
    DB_USER = os.getenv('DB_USER', 'collision_admin')      # adjust if your user is different
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'CollisionDB2024!')  # your actual password

    DATABASE_URL = (
        f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    # Other app settings if needed
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'