import os, subprocess, sys

def main():
    # Simple helper that launches the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
