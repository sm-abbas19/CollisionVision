import psycopg2
from psycopg2.extras import RealDictCursor
import threading
from CollisionVision.config.config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.connection_params = {
            'host': Config.DB_HOST,
            'port': Config.DB_PORT,
            'database': Config.DB_NAME,
            'user': Config.DB_USER,
            'password': Config.DB_PASSWORD
        }
        self.lock = threading.Lock()
        logger.info("DatabaseManager initialized")

    def get_connection(self):
        """Create a new database connection"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def log_collision_event(self, event_data):
        """Insert collision event into database"""
        try:
            with self.lock:
                conn = self.get_connection()
                cursor = conn.cursor()

                insert_query = """
                               INSERT INTO collision_events
                               (event_type, probability, distance, object1_id, object2_id,
                                object1_class, object2_class, frame_number, severity)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id; \
                               """

                cursor.execute(insert_query, (
                    event_data.get('event_type'),
                    event_data.get('probability'),
                    event_data.get('distance'),
                    event_data.get('object1_id'),
                    event_data.get('object2_id'),
                    event_data.get('object1_class'),
                    event_data.get('object2_class'),
                    event_data.get('frame_number'),
                    event_data.get('severity')
                ))

                event_id = cursor.fetchone()[0]
                conn.commit()
                cursor.close()
                conn.close()

                logger.info(f"Collision event logged with ID: {event_id}")
                return event_id

        except psycopg2.Error as e:
            logger.error(f"Error logging collision event: {e}")
            raise

    def log_system_metrics(self, metrics_data):
        """Insert system metrics into database"""
        try:
            with self.lock:
                conn = self.get_connection()
                cursor = conn.cursor()

                insert_query = """
                               INSERT INTO system_metrics
                                   (fps, total_objects_detected, active_tracks, memory_usage, cpu_usage)
                               VALUES (%s, %s, %s, %s, %s) RETURNING id; \
                               """

                cursor.execute(insert_query, (
                    metrics_data.get('fps'),
                    metrics_data.get('total_objects_detected'),
                    metrics_data.get('active_tracks'),
                    metrics_data.get('memory_usage'),
                    metrics_data.get('cpu_usage')
                ))

                metrics_id = cursor.fetchone()[0]
                conn.commit()
                cursor.close()
                conn.close()

                logger.info(f"System metrics logged with ID: {metrics_id}")
                return metrics_id

        except psycopg2.Error as e:
            logger.error(f"Error logging system metrics: {e}")
            raise

    def get_recent_events(self, limit=100, table_name="collision_events"):
        """
        Retrieve recent records from the specified table.
        By default, it fetches from collision_events.
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Ensure only allowed tables can be queried for safety
            allowed_tables = ["collision_events", "system_metrics"]
            if table_name not in allowed_tables:
                raise ValueError(f"Table {table_name} not allowed.")

            query = f"""
                    SELECT * 
                    FROM {table_name}
                    ORDER BY timestamp DESC
                    LIMIT %s;
                    """

            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

            cursor.close()
            conn.close()

            return [dict(row) for row in rows]

        except psycopg2.Error as e:
            logger.error(f"Error retrieving rows from {table_name}: {e}")
            raise
        except Exception as ex:
            logger.error(f"Error: {ex}")
            raise

    def get_event_statistics(self):
        """Get summary statistics for events"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            query = """
                    SELECT COUNT(*)                                             as total_events, \
                           COUNT(CASE WHEN event_type = 'collision' THEN 1 END) as collisions, \
                           COUNT(CASE WHEN event_type = 'near_miss' THEN 1 END) as near_misses, \
                           COUNT(CASE WHEN event_type = 'warning' THEN 1 END)   as warnings, \
                           COUNT(CASE WHEN resolved = FALSE THEN 1 END)         as unresolved, \
                           AVG(probability)                                     as avg_probability, \
                           AVG(distance)                                        as avg_distance
                    FROM collision_events
                    WHERE timestamp >= CURRENT_DATE; \
                    """

            cursor.execute(query)
            stats = cursor.fetchone()

            cursor.close()
            conn.close()

            return dict(stats) if stats else {}

        except psycopg2.Error as e:
            logger.error(f"Error retrieving statistics: {e}")
            raise

    def test_connection(self):
        """Test database connection"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.close()
            conn.close()
            logger.info("Database connection test successful")
            return True
        except psycopg2.Error as e:
            logger.error(f"Database connection test failed: {e}")
            return False