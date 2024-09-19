import subprocess
import os

def start_streamlit():
    """
    Startet die Streamlit App fashion.py
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir,"fashion.py")
    subprocess.run(["streamlit","run",script_path])

if __name__ == "__main__":
    start_streamlit()