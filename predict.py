from rag_recommender import KnowledgeBaseIndex, generate_recommendation
from joblib import load
from tkinter import messagebox
import tkinter as tk
import pandas as pd
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load saved model and encoders
model = load('mental_health_model.joblib')
le_dict = load('label_encoders.joblib')

print("Building RAG knowledge base index...")
kb_index = KnowledgeBaseIndex()

features = ['Age', 'Gender', 'Daily Screen Time (hours)',
            'Do you have Anxiety?', 'Do you have Panic attack?',
            'Sleep Schedule', 'Exercise Frequency',
            'Family History of Mental Illness',
            'Substance Use', 'Stress Levels', 'Work-Life Balance',
            'Mood', 'mental_health_score']


def predict():
    try:
        gender = le_dict['Gender'].transform([entry_gender.get()])[0]
        sleep = le_dict['Sleep Schedule'].transform([entry_sleep.get()])[0]
        exercise = le_dict['Exercise Frequency'].transform(
            [entry_exercise.get()])[0]
        substance = le_dict['Substance Use'].transform(
            [entry_substance.get()])[0]
        mood = le_dict['Mood'].transform([entry_mood.get()])[0]

        anxiety = int(var_anxiety.get())
        panic = int(var_panic.get())
        stress = int(entry_stress.get())

        mental_health_score = anxiety + panic + stress / 10

        # Encoded dict — used for the ML model, values are numeric codes
        user_data = {
            'Age': float(entry_age.get()),
            'Gender': gender,
            'Daily Screen Time (hours)': float(entry_screen.get()),
            'Do you have Anxiety?': anxiety,
            'Do you have Panic attack?': panic,
            'Sleep Schedule': sleep,
            'Exercise Frequency': exercise,
            'Family History of Mental Illness': int(var_family.get()),
            'Substance Use': substance,
            'Stress Levels': stress,
            'Work-Life Balance': int(entry_wlb.get()),
            'Mood': mood,
            'mental_health_score': mental_health_score
        }

        # Raw dict — original text values, used only for the RAG risk-factor
        # lookup, so it doesn't get confused by label-encoded numbers.
        raw_user_data = {
            'Do you have Anxiety?': anxiety,
            'Do you have Panic attack?': panic,
            'Sleep Schedule': entry_sleep.get(),
            'Exercise Frequency': entry_exercise.get(),
            'Family History of Mental Illness': int(var_family.get()),
            'Substance Use': entry_substance.get(),
            'Work-Life Balance': int(entry_wlb.get()),
            'Daily Screen Time (hours)': float(entry_screen.get()),
        }

        df = pd.DataFrame([user_data])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        suggestion, retrieved, factors = generate_recommendation(
            kb_index, raw_user_data, pred, prob
        )

        result = "⚠️ Depression Likely" if pred == 1 else "✅ No Depression Detected"
        messagebox.showinfo(
            "Result",
            f"{result}\nConfidence: {prob*100:.1f}%\n\nSuggestion:\n{suggestion}"
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))


# UI
win = tk.Tk()
win.title("Depression Status Predictor")
win.geometry("400x700")
win.configure(bg="#1e1e1e")


def lbl(text):
    tk.Label(win, text=text, bg="#1e1e1e", fg="white",
             font=("Helvetica", 10)).pack(pady=(8, 0))


def entry():
    e = tk.Entry(win, width=30)
    e.pack()
    return e


lbl("Age")
entry_age = entry()

lbl("Gender (Female/Male/Other)")
entry_gender = entry()

lbl("Daily Screen Time (hours)")
entry_screen = entry()

lbl("Anxiety? (1=Yes 0=No)")
var_anxiety = tk.StringVar(value="0")
tk.OptionMenu(win, var_anxiety, "0", "1").pack()

lbl("Panic Attack? (1=Yes 0=No)")
var_panic = tk.StringVar(value="0")
tk.OptionMenu(win, var_panic, "0", "1").pack()

lbl("Sleep Schedule (Regular/Irregular)")
entry_sleep = entry()

lbl("Exercise Frequency (Regular/Occasional/Never)")
entry_exercise = entry()

lbl("Family History of Mental Illness (1=Yes 0=No)")
var_family = tk.StringVar(value="0")
tk.OptionMenu(win, var_family, "0", "1").pack()

lbl("Substance Use (None/Occasional/Frequent)")
entry_substance = entry()

lbl("Stress Levels (1-10)")
entry_stress = entry()

lbl("Work-Life Balance (1-10)")
entry_wlb = entry()

lbl("Mood (Happy/Neutral/Sad)")
entry_mood = entry()

tk.Button(win, text="Predict", command=predict,
          bg="#4CAF50", fg="white", font=("Helvetica", 12),
          width=20, height=2).pack(pady=20)

win.mainloop()
