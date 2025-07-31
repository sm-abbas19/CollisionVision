from CollisionVision.database.database import DatabaseManager

if __name__ == "__main__":
    db = DatabaseManager()
    if db.test_connection():
        print("Database connection successful!")
    else:
        print("Database connection failed.")