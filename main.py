import subprocess
import os


def run_training():
    python_path = os.path.join('venv', 'Scripts', 'python.exe')
    subprocess.run([python_path, 'src/train.py'])

if __name__ == "__main__":
    run_training()
