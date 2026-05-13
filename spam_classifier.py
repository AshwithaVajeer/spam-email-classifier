import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ----------------------------
# Sample Dataset
# ----------------------------
data = {
    "message": [
        "Congratulations! You won a free iPhone",
        "Claim your cash prize now",
        "Win money instantly click here",
        "Free lottery ticket available",
        "Meeting at 3 PM tomorrow",
        "Project submission deadline is Friday",
        "Please review the attached report",
        "Let's have lunch tomorrow",
        "Can you send the assignment",
        "Reminder for tomorrow's class"
    ],
    "label": [
        "spam",
        "spam",
        "spam",
        "spam",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham"
    ]
}

df = pd.DataFrame(data)

# ----------------------------
# Model Training
# ----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["message"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

# ----------------------------
# GUI Functions
# ----------------------------
def classify_message():
    msg = text_box.get("1.0", tk.END).strip()

    if msg == "":
        messagebox.showwarning("Warning", "Please enter a message")
        return

    transformed = vectorizer.transform([msg])
    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed).max() * 100

    if prediction == "spam":
        result_label.config(
            text=f"Result: SPAM ({probability:.2f}%)",
            fg="red"
        )
    else:
        result_label.config(
            text=f"Result: NOT SPAM ({probability:.2f}%)",
            fg="green"
        )


def clear_text():
    text_box.delete("1.0", tk.END)
    result_label.config(
        text="Enter an email/message to classify",
        fg="black"
    )


# ----------------------------
# GUI Window
# ----------------------------
root = tk.Tk()
root.title("Spam Email Classifier")
root.geometry("550x500")
root.configure(bg="#f4f6f8")
root.resizable(False, False)

title = tk.Label(
    root,
    text="Spam Email Classifier",
    font=("Arial", 22, "bold"),
    bg="#f4f6f8"
)
title.pack(pady=15)

instruction = tk.Label(
    root,
    text="Type or paste an email/message below:",
    font=("Arial", 12),
    bg="#f4f6f8"
)
instruction.pack()

text_box = tk.Text(
    root,
    height=10,
    width=55,
    font=("Arial", 12),
    relief="solid",
    bd=2
)
text_box.pack(pady=15)

btn_frame = tk.Frame(root, bg="#f4f6f8")
btn_frame.pack()

check_btn = tk.Button(
    btn_frame,
    text="Check",
    command=classify_message,
    bg="#28a745",
    fg="white",
    font=("Arial", 13, "bold"),
    width=12
)
check_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(
    btn_frame,
    text="Clear",
    command=clear_text,
    bg="#dc3545",
    fg="white",
    font=("Arial", 13, "bold"),
    width=12
)
clear_btn.grid(row=0, column=1, padx=10)

result_label = tk.Label(
    root,
    text="Enter an email/message to classify",
    font=("Arial", 16, "bold"),
    bg="#f4f6f8"
)
result_label.pack(pady=30)

root.mainloop()