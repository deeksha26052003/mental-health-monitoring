import tkinter as tk
from tkinter import messagebox
import subprocess
from PIL import Image, ImageTk
import requests
from io import BytesIO


def load_image_from_web(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        return Image.open(image_data)
    else:
        raise Exception(
            f"Failed to load image from {url}. HTTP Status Code: {response.status_code}")


def create_main_menu():
    window = tk.Tk()
    window.title("Mental Health Monitoring System")

    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    window.geometry(f"{screen_width}x{screen_height}")

    try:
        url = "https://tse4.mm.bing.net/th?id=OIP.6CYDBjIp1G1G-eQZkdAYfgHaEc&pid=Api&P=0&h=180"
        bg_image = load_image_from_web(url)
        bg_photo = ImageTk.PhotoImage(bg_image.resize(
            (screen_width, screen_height), Image.Resampling.LANCZOS))
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {str(e)}")
        return

    label_bg = tk.Label(window, image=bg_photo)
    label_bg.place(x=0, y=0, relwidth=1, relheight=1)

    frame_buttons = tk.Frame(window, bg="black", bd=0)
    frame_buttons.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(
        frame_buttons,
        text="Mental Health Monitoring System",
        font=("Helvetica", 18, "bold"),
        fg="white",
        bg="black"
    ).pack(pady=40)

    tk.Label(
        frame_buttons,
        text="Please Select A Model to Run",
        font=("Helvetica", 18, "bold"),
        fg="white",
        bg="black"
    ).pack(pady=20)

    button_style = {
        'width': 25,
        'height': 2,
        'font': ("Helvetica", 12),
        'bd': 0,
        'relief': "solid",
        'bg': "#4CAF50",
        'fg': "white",
        'activebackground': "#45a049",
        'activeforeground': "white"
    }

    tk.Button(frame_buttons, text="Depression Status Predictor",
              command=lambda: subprocess.run(["python3", "predict.py"]),
              **button_style).pack(pady=15)
    tk.Button(frame_buttons, text="Respiration Monitoring",
              command=lambda: subprocess.run(["python3", "respiration.py"]),
              **button_style).pack(pady=15)
    tk.Button(frame_buttons, text="Facial Expression Detection",
              command=lambda: subprocess.run(
                  ["python3", "facial_expression.py"]),
              **button_style).pack(pady=15)
    tk.Button(frame_buttons, text="Body Temperature Detection",
              command=lambda: subprocess.run(
                  ["python3", "body_temperature.py"]),
              **button_style).pack(pady=15)

    tk.Label(window, text="© 2024 Mental Health Project | All rights reserved.",
             font=("Helvetica", 8), fg="#777", bg="black").pack(side="bottom", pady=5)

    label_bg.image = bg_photo
    window.mainloop()


if __name__ == "__main__":
    create_main_menu()
