import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

def fetch_section_results(workout_result_id):
    # Get DB configuration from environment variables
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB")

    if not db_user or not db_password or not db_name:
        print("Error: DB_USER, DB_PASSWORD, and DB_NAME must be set in environment variables.")
        return

    try:
        # Connect to Postgres
        conn = psycopg2.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            database=db_name
        )
        
        # Use RealDictCursor to get results as dictionaries
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Query section results for the given workout_result_id
        # We assume section_results table has a workout_result_id column
        query = "SELECT * FROM section_results WHERE workout_result_id = %s"
        cur.execute(query, (workout_result_id,))
        results = cur.fetchall()

        if not results:
            print(f"No section results found for workout result ID: {workout_result_id}")
            return

        # Create output directory
        output_dir = str(workout_result_id)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing {len(results)} section results for workout result ID {workout_result_id} into directory '{output_dir}/'...")

        for idx, row in enumerate(results):
            # Each section result will be saved under the workout_result_id directory.
            # We use a section identifier to name the files if they are multiple sections.
            section_id = row.get("id", f"section_{idx}")
            section_subdir = os.path.join(output_dir, f"section_{section_id}")
            os.makedirs(section_subdir, exist_ok=True)
            
            # 1. Extract and save the "data" field (JSON)
            data_field = row.get("data")
            if data_field:
                # Based on the description "Let's create a file with it." 
                data_path = os.path.join(section_subdir, "data.json")
                with open(data_path, "w", encoding="utf-8") as f:
                    json.dump(data_field, f, indent=4)
                print(f"  -> Saved data for section {section_id} to {data_path}")
            else:
                print(f"  -> Warning: No 'data' field found for section {section_id}")

            # 2. Create a "section_data" file (also a JSON)
            # "Let's also create a file 'section_data' (also a json)"
            section_data_path = os.path.join(section_subdir, "section_data.json")
            # Usually section_data refers to the whole row in this context
            section_info = {k: v for k, v in row.items() if k != "data"}
            with open(section_data_path, "w", encoding="utf-8") as f:
                json.dump(section_info, f, indent=4, default=str)
            print(f"  -> Saved section_data for section {section_id} to {section_data_path}")

        cur.close()
        conn.close()
        print("Done.")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Fetch section results for a workout result ID from Postgres.")
    # parser.add_argument("workout_result_id", help="The ID of the workout result.")
    # args = parser.parse_args()

    fetch_section_results(6123)
