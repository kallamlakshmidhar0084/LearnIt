import os
import psycopg
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DBClient:
    def __init__(self):
        # Read credentials from environment
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.user = os.getenv("DB_USER", "lkallam")
        self.password = os.getenv("DB_PASSWORD", "password")
        self.dbname = os.getenv("DB_NAME", "my_database")
        
        self.conn_str = f"host={self.host} port={self.port} user={self.user} password={self.password} dbname={self.dbname}"

    def get_connection(self):
        """Returns a new connection object."""
        try:
            conn = psycopg.connect(self.conn_str)
            return conn
        except psycopg.Error as e:
            print(f"Error connecting to database: {e}")
            return None
    
    def execute_query(self, query: str, params: tuple = ()):
        """Executes a query and returns results."""
        with self.get_connection() as conn:
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute(query, params)
                
                # Commit if it's a modification query
                if query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
                    conn.commit()
                    return cur.rowcount
                
                # Fetch results if it's a SELECT query
                try:
                    return cur.fetchall()
                except psycopg.ProgrammingError:
                    return None

# Example usage
if __name__ == "__main__":
    client = DBClient()
    
    # Test connection and create a sample table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        email VARCHAR(100) UNIQUE
    );
    """
    client.execute_query(create_table_query)
    print("Table created or already exists.")

    # Insert a record
    client.execute_query(
        "INSERT INTO users (name, email) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        ("John Doe", "john@example.com")
    )

    # Read records
    rows = client.execute_query("SELECT * FROM users;")
    print("Users in database:", rows)