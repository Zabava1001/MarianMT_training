import subprocess
import os
import platform


def get_python_path():
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "python.exe")
    else:
        return os.path.join("venv", "bin", "python")


def run_script(choice):
    python_path = get_python_path()

    if choice == "1":
        script = "src/train.py"
    elif choice == "2":
        script = "bleu/bleu_score.py"
    elif choice == "3":
        script = "generate_data/gen_data.py"
    else:
        print("Неверный выбор. Доступны только: 1, 2 или 3.")
        return

    print(f"Запуск: {script}")
    subprocess.run([python_path, script])


if __name__ == "__main__":
    print("Что запустить?")
    print("1 — Обучение модели (train.py)")
    print("2 — Расчёт BLEU (bleu_score.py)")
    print("3 — Генерация синтетических данных (gen_data.py)")

    user_choice = input("Выберите (1/2/3): ").strip()
    run_script(user_choice)
