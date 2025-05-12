import os
import subprocess
import sys

# Determine the path to the frontend_pipeline.py file
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "App", "frontend_pipeline.py")

def main():
    print("Starting CV Classification Pipeline Frontend...")
    print(f"Frontend path: {FRONTEND_PATH}")
    
    # Run the Streamlit app
    try:
        subprocess.run([
            "streamlit", "run", 
            FRONTEND_PATH,
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Streamlit not found. Please install it with:")
        print("pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main() 