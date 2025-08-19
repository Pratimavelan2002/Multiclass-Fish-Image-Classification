import os
import subprocess
import sys

def main():
    
    cmd = [sys.executable, "-m", "streamlit", "run", "fish_classifier_app.py"]
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
