# main_app_windows.py
"""
AI Smart Healthcare (separate animated windows)
- Uses customtkinter for modern look
- Each major screen opens as a separate Toplevel window with fade-in/out
- Multi-disease: Diabetes, Heart, Cancer (loads diabetes.pkl, heart.pkl, cancer.pkl if present)
- Precautions & Advice on High Risk
- Doctor & Patient dashboards, Admin model upload
- SQLite storage: users + predictions (input JSON + advice)
- Notification helpers (email / SMS) are placeholders — add credentials to enable
"""
from chat_backend import init_chat_db, send_message, get_messages
from chatbot_ai import ai_response
import os
import sqlite3

def add_message(frame, text, sender="user"):
    bubble = ctk.CTkFrame(
        frame,
        corner_radius=15,
        fg_color="#158FBE" if sender == "user" else "#1789E7"
    )

    label = ctk.CTkLabel(
        bubble,
        text=text,
        wraplength=280,
        justify="left"
    )
    label.pack(padx=10, pady=5)

    if sender == "user":
        bubble.pack(anchor="e", padx=10, pady=3)
    else:
        bubble.pack(anchor="w", padx=10, pady=3)

init_chat_db()
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ctk
import threading
import time
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")



# ---------- Configuration ----------
DB_FILE = "healthcare_windows.db"
# Notification config (fill to enable)
SMTP_EMAIL = ""
SMTP_PASSWORD = ""
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
TWILIO_SID = ""
TWILIO_AUTH = ""
TWILIO_PHONE = ""

# ---------- Fallback tiny models (train if missing) ----------
def train_fallback_models():
    if not os.path.exists("diabetes.pkl"):
        df = pd.DataFrame([
            [2,120,70,32.0,0],
            [4,150,80,33.5,1],
            [1,85,66,26.6,0],
            [3,180,90,34.2,1],
            [0,100,76,25.3,0],
            [5,140,82,35.4,1],
            [2,90,70,28.5,0],
            [6,160,85,36.2,1],
            [1,95,60,23.1,0],
            [4,170,88,37.1,1]
        ], columns=["Pregnancies","Glucose","BloodPressure","BMI","Outcome"])
        X = df[["Pregnancies","Glucose","BloodPressure","BMI"]]; y = df["Outcome"]
        m = LogisticRegression(max_iter=200).fit(X,y)
        with open("diabetes.pkl","wb") as f: pickle.dump(m,f)
    if not os.path.exists("heart.pkl"):
        df = pd.DataFrame([
            [45,130,230,150,1],
            [54,140,250,120,1],
            [34,120,180,170,0],
            [60,150,260,110,1],
            [29,110,170,180,0],
            [50,145,240,130,1],
            [40,125,200,160,0],
            [58,155,270,115,1],
            [36,118,175,172,0],
            [48,135,220,148,1]
        ], columns=["Age","RestingBP","Cholesterol","MaxHR","Outcome"])
        X = df[["Age","RestingBP","Cholesterol","MaxHR"]]; y = df["Outcome"]
        m = LogisticRegression(max_iter=200).fit(X,y)
        with open("heart.pkl","wb") as f: pickle.dump(m,f)
    if not os.path.exists("cancer.pkl"):
        df = pd.DataFrame([
            [50,2.1,1,0.2,0],
            [62,4.5,3,1.8,1],
            [35,1.0,0,0.1,0],
            [70,5.2,4,2.0,1],
            [28,0.8,0,0.05,0],
            [58,3.8,2,1.0,1],
            [42,1.5,0,0.3,0],
            [65,4.9,3,1.7,1],
            [31,0.9,0,0.07,0],
            [55,3.2,1,0.9,1]
        ], columns=["Age","TumorSize","LumpCount","Biomarker","Outcome"])
        X = df[["Age","TumorSize","LumpCount","Biomarker"]]; y = df["Outcome"]
        m = LogisticRegression(max_iter=200).fit(X,y)
        with open("cancer.pkl","wb") as f: pickle.dump(m,f)

train_fallback_models()

# ---------- Load models ----------
def load_model_file(path):
    try:
        with open(path,"rb") as f: return pickle.load(f)
    except Exception as e:
        print("Model load failed:", path, e)
        return None

MODELS = {
    "Diabetes": load_model_file("diabetes.pkl"),
    "Heart": load_model_file("heart.pkl"),
    "Cancer": load_model_file("cancer.pkl")
}

# ---------- Advice text ----------
ADVICE = {
    "Diabetes": [
        "Avoid sugary foods & drinks.",
        "Exercise 30 mins daily.",
        "Monitor blood glucose regularly.",
        "Consult your physician for tests."
    ],
    "Heart": [
        "Reduce salt & saturated fats.",
        "Do cardio exercise regularly.",
        "Avoid smoking and limit alcohol.",
        "Seek cardiology consult if symptoms."
    ],
    "Cancer": [
        "Schedule follow-up imaging/biopsy urgently.",
        "Avoid tobacco and limit alcohol.",
        "Seek specialist referral.",
        "Keep a symptom diary and inform doctor."
    ]
}

# ---------- Database ----------
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT,
    email TEXT,
    phone TEXT
)""")
cur.execute("""CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    disease TEXT,
    input_json TEXT,
    result TEXT,
    advice TEXT,
    timestamp TEXT
)""")
conn.commit()

# seed doctor
try:
    cur.execute("INSERT INTO users(username,password,role,email) VALUES(?,?,?,?)",
                ("doctor","doc123","doctor","doctor@example.com"))
    conn.commit()
except:
    pass

# ---------- Notification placeholders ----------
def send_email(to_email, subject, body):
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print("Email disabled (set SMTP_EMAIL/SMTP_PASSWORD).")
        return False
    try:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_EMAIL
        msg["To"] = to_email
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.login(SMTP_EMAIL, SMTP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print("Email send failed:", e)
        return False

def send_sms(to_number, message):
    if not (TWILIO_SID and TWILIO_AUTH and TWILIO_PHONE):
        print("SMS disabled (Twilio creds missing).")
        return False
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_AUTH)
        client.messages.create(body=message, from_=TWILIO_PHONE, to=to_number)
        return True
    except Exception as e:
        print("SMS failed:", e)
        return False

# ---------- DB helper functions ----------
def register_user_db(username,password,role,email="",phone=""):
    try:
        cur.execute("INSERT INTO users(username,password,role,email,phone) VALUES(?,?,?,?,?)",
                    (username,password,role,email,phone))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def validate_login(username,password):
    cur.execute("SELECT role,email,phone FROM users WHERE username=? AND password=?", (username,password))
    r = cur.fetchone()
    if r:
        return {"role": r[0], "email": r[1], "phone": r[2]}
    return None

def save_prediction_db(username,disease,input_dict,result,advice):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO predictions(username,disease,input_json,result,advice,timestamp) VALUES(?,?,?,?,?,?)",
                (username,disease,json.dumps(input_dict),result,advice,ts))
    conn.commit()

def fetch_user_predictions(username):
    cur.execute("SELECT id,disease,input_json,result,advice,timestamp FROM predictions WHERE username=? ORDER BY id DESC", (username,))
    return cur.fetchall()

def fetch_all_predictions(user_filter=None, disease_filter=None, from_date=None, to_date=None):
    q = "SELECT id,username,disease,input_json,result,advice,timestamp FROM predictions WHERE 1=1"
    params = []
    if user_filter:
        q += " AND username LIKE ?"; params.append('%'+user_filter+'%')
    if disease_filter:
        q += " AND disease=?"; params.append(disease_filter)
    if from_date:
        q += " AND date(timestamp) >= date(?)"; params.append(from_date)
    if to_date:
        q += " AND date(timestamp) <= date(?)"; params.append(to_date)
    q += " ORDER BY id DESC"
    cur.execute(q, params)
    return cur.fetchall()

# ---------- UI helpers: fade in/out for Toplevel windows ----------
def fade_in(win, step=0.06, delay=0.01):
    try:
        win.attributes("-alpha", 0.0)
        a = 0.0
        while a < 1.0:
            a = round(a + step, 2)
            if a > 1.0: a = 1.0
            win.attributes("-alpha", a)
            win.update()
            time.sleep(delay)
    except Exception:
        pass

def fade_out_and_destroy(win, step=0.06, delay=0.01):
    try:
        a = win.attributes("-alpha")
        if a is None: a = 1.0
        while a > 0:
            a = round(a - step, 2)
            if a < 0: a = 0.0
            win.attributes("-alpha", a)
            win.update()
            time.sleep(delay)
    except Exception:
        pass
    try:
        win.destroy()
    except:
        pass

# ---------- CustomTK App root ----------
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
ROOT = ctk.CTk()
ROOT.title("AI Smart Healthcare System")
ROOT.geometry("1100x650")
ROOT.configure(fg_color="#F4F6F8")

header = ctk.CTkFrame(ROOT, height=60, fg_color="#1E88E5")
header.pack(fill="x")

ctk.CTkLabel(
    header,
    text="AI Smart Healthcare System",
    text_color="white",
    font=("Arial", 22, "bold")
).pack(pady=12)


# ---------- Launch window (main) ----------
def open_login_window():
    # open login as separate Toplevel with fade
    win = ctk.CTkToplevel(ROOT)
    win.title("Login")
    win.geometry("480x420")
    win.resizable(False, False)

    # UI
    lbl = ctk.CTkLabel(win, text="Smart Healthcare Login", font=("Segoe UI", 18, "bold"))
    lbl.pack(pady=18)
    frame = ctk.CTkFrame(
    ROOT,
    corner_radius=15,
    fg_color="#FFFFFF"
)
    frame.pack(padx=20, pady=20, fill="both", expand=True)

    tk.Label(frame, text="Username").pack(pady=(10,2))
    et_user = ctk.CTkEntry(frame); et_user.pack(pady=4)
    tk.Label(frame, text="Password").pack(pady=(10,2))
    et_pass = ctk.CTkEntry(frame, show="*"); et_pass.pack(pady=4)

    def do_login():
        u = et_user.get().strip(); p = et_pass.get().strip()
        if not u or not p:
            messagebox.showerror("Input error","Enter username and password"); return
        info = validate_login(u,p)
        if not info:
            messagebox.showerror("Login failed","Invalid credentials"); return
        # fade out login and open next window
        fade_out_and_destroy(win)
        if info["role"] == "patient":
            open_patient_window(u, info["email"], info["phone"])
        else:
            open_doctor_window()

    btn_login = ctk.CTkButton(frame, text="Login", command=do_login)
    btn_login.pack(pady=12)
    btn_register = ctk.CTkButton(frame, text="Register", fg_color="#16a085", command=lambda: (fade_out_and_destroy(win), open_register_window()))
    btn_register.pack(pady=6)

    # animate appear
    win.attributes("-alpha",0.0)
    win.update()
    fade_in(win)

def open_register_window():
    win = ctk.CTkToplevel(ROOT)
    win.title("Register")
    win.geometry("520x520")
    win.resizable(False, False)

    lbl = ctk.CTkLabel(win, text="Create Account", font=("Segoe UI", 18, "bold"))
    lbl.pack(pady=12)
    frame = ctk.CTkFrame(win); frame.pack(padx=16, pady=8, fill="both", expand=True)

    tk.Label(frame, text="Username").pack(pady=(8,2))
    et_user = ctk.CTkEntry(frame); et_user.pack(pady=4)
    tk.Label(frame, text="Password").pack(pady=(8,2))
    et_pass = ctk.CTkEntry(frame, show="*"); et_pass.pack(pady=4)
    tk.Label(frame, text="Role").pack(pady=(8,2))
    role_cb = ttk.Combobox(frame, values=["patient","doctor"], state="readonly"); role_cb.set("patient"); role_cb.pack(pady=4)
    tk.Label(frame, text="Email (optional)").pack(pady=(8,2))
    et_email = ctk.CTkEntry(frame); et_email.pack(pady=4)
    tk.Label(frame, text="Phone (optional)").pack(pady=(8,2))
    et_phone = ctk.CTkEntry(frame); et_phone.pack(pady=4)

    def do_register():
        u = et_user.get().strip(); p = et_pass.get().strip(); r = role_cb.get(); em = et_email.get().strip(); ph = et_phone.get().strip()
        if not u or not p:
            messagebox.showerror("Error","Username & password required"); return
        ok = register_user_db(u,p,r,em,ph)
        if ok:
            messagebox.showinfo("Success","Account created") 
            fade_out_and_destroy(win)
            open_login_window()
        else:
            messagebox.showerror("Error","Username exists")

    ctk.CTkButton(
    frame,
    text="Predict Disease",
    fg_color="#1E88E5",
    hover_color="#1565C0",
    text_color="white",
    corner_radius=10
).pack(pady=6)

    ctk.CTkButton(
    frame,
    text="AI Health Chatbot",
    fg_color="#43A047",
    hover_color="#2E7D32"
).pack(pady=6)

    win.attributes("-alpha",0.0); win.update(); fade_in(win)

def open_patient_chat(username):
    win = ctk.CTkToplevel(ROOT)
    win.title("Chat with Doctor")
    win.geometry("420x550")

    chat_area = ctk.CTkScrollableFrame(win)
    chat_area.pack(fill="both", expand=True, padx=5, pady=5)

    input_frame = ctk.CTkFrame(win)
    input_frame.pack(fill="x", padx=5, pady=5)

    entry = ctk.CTkEntry(input_frame, placeholder_text="Type message...")
    entry.pack(side="left", fill="x", expand=True, padx=5)

    def refresh_chat():
        for w in chat_area.winfo_children():
            w.destroy()

        msgs = get_messages(username, "doctor")
        for sender, msg, _ in msgs:
            if sender == username:
                add_message(chat_area, msg, "user")
            else:
                add_message(chat_area, msg, "bot")

    def send_chat():
        msg = entry.get().strip()
        if msg:
            send_message(username, "doctor", msg)
            entry.delete(0, "end")
            refresh_chat()

    ctk.CTkButton(input_frame, text="Send", command=send_chat).pack(side="right", padx=5)
    refresh_chat()


def open_ai_chatbot(disease=None):
    win = ctk.CTkToplevel(ROOT)
    win.title("AI Health Assistant")
    win.geometry("420x550")

    # Chat area (scrollable)
    chat_area = ctk.CTkScrollableFrame(win)
    chat_area.pack(fill="both", expand=True, padx=5, pady=5)

    # Input area
    input_frame = ctk.CTkFrame(win)
    input_frame.pack(fill="x", padx=5, pady=5)

    entry = ctk.CTkEntry(input_frame, placeholder_text="Type a message...")
    entry.pack(side="left", fill="x", expand=True, padx=5)

    def send_msg():
        msg = entry.get().strip()
        if not msg:
            return

        add_message(chat_area, msg, "user")
        reply = ai_response(msg, disease)
        add_message(chat_area, reply, "bot")

        entry.delete(0, "end")

    ctk.CTkButton(
        input_frame,
        text="Send",
        width=70,
        command=send_msg
    ).pack(side="right", padx=5)



def open_doctor_chat():
    win = ctk.CTkToplevel(ROOT)
    win.title("Chat with Patient")
    win.geometry("500x500")

    chat_box = tk.Text(win, wrap="word")
    chat_box.pack(fill="both", expand=True, padx=10, pady=10)

    entry = ctk.CTkEntry(win)
    entry.pack(fill="x", padx=10, pady=5)

    def refresh_chat():
        chat_box.delete("1.0", "end")
        msgs = get_messages("patient", "doctor")
        for s, m, t in msgs:
            chat_box.insert("end", f"{s}: {m}\n")

    def send_chat():
        msg = entry.get().strip()
        if msg:
            send_message("doctor", "patient", msg)
            entry.delete(0, "end")
            refresh_chat()

    btn_frame = ctk.CTkFrame(win)
    btn_frame.pack(pady=5)
    ctk.CTkButton(btn_frame, text="Reply", command=send_chat).pack(side="left", padx=5)
    ctk.CTkButton(btn_frame, text="Refresh", command=refresh_chat).pack(side="left", padx=5)

    refresh_chat()


# ---------- Patient window ----------
def open_patient_window(username, email, phone):
    win = ctk.CTkToplevel(ROOT); win.title(f"Patient - {username}"); win.geometry("1000x700"); win.minsize(900,650)
    # left form, right history + advice
    left = ctk.CTkFrame(win); left.place(relx=0.02, rely=0.02, relwidth=0.46, relheight=0.96)
    right = ctk.CTkFrame(win); right.place(relx=0.50, rely=0.02, relwidth=0.48, relheight=0.96)
    ctk.CTkLabel(left, text=f"Hello, {username}", font=("Segoe UI", 16, "bold")).pack(pady=10)
    ctk.CTkLabel(left, text="Select Disease").pack(pady=(6,2))
    disease_var = tk.StringVar(value="Diabetes")
    disease_cb = ttk.Combobox(left, values=["Diabetes","Heart","Cancer"], state="readonly"); disease_cb.set("Diabetes"); disease_cb.pack(pady=4)

    # Define field specs per disease
    specs = {
        "Diabetes":[("Pregnancies","int"),("Glucose","int"),("BloodPressure","int"),("BMI","float")],
        "Heart":[("Age","int"),("RestingBP","int"),("Cholesterol","int"),("MaxHR","int")],
        "Cancer":[("Age","int"),("TumorSize_cm","float"),("LumpCount","int"),("BiomarkerValue","float")]
    }
    field_entries = {}
    fields_frame = ctk.CTkScrollableFrame(left, height=300); fields_frame.pack(fill="x", padx=6, pady=8)

    def build_fields(disease):
        for w in fields_frame.winfo_children(): w.destroy()
        field_entries.clear()
        for label,typ in specs[disease]:
            tk.Label(fields_frame, text=label).pack(anchor="w", pady=(6,2))
            e = ctk.CTkEntry(fields_frame); e.pack(fill="x", pady=(0,4))
            field_entries[label] = (e, typ)
    build_fields("Diabetes")
    disease_cb.bind("<<ComboboxSelected>>", lambda e: build_fields(disease_cb.get()))

    # right side: result + advice + history
    lbl_res = ctk.CTkLabel(right, text="Prediction Result", font=("Segoe UI", 14, "bold")); lbl_res.pack(pady=8)
    res_var = tk.StringVar(); res_label = ctk.CTkLabel(right, textvariable=res_var, font=("Segoe UI", 13)); res_label.pack(pady=6)
    ctk.CTkLabel(right, text="Precautions / Next Steps", font=("Segoe UI", 12, "bold")).pack(pady=(8,2))
    advice_box = tk.Text(right, height=10, wrap="word"); advice_box.pack(fill="both", padx=8, pady=6)

    notify_email = tk.BooleanVar(value=False); notify_sms = tk.BooleanVar(value=False)
    ctk.CTkCheckBox(right, text="Send Email (if set)", variable=notify_email).pack(anchor="w", padx=8)
    ctk.CTkCheckBox(right, text="Send SMS (if set)", variable=notify_sms).pack(anchor="w", padx=8)

    # history tree
    hist_frame = ctk.CTkFrame(right); hist_frame.pack(fill="both", expand=True, padx=6, pady=6)
    cols = ("ID","Disease","Result","Time")
    tree = ttk.Treeview(hist_frame, columns=cols, show="headings", height=8)
    for c in cols:
        tree.heading(c, text=c); tree.column(c, anchor="center")
    vsb = ttk.Scrollbar(hist_frame, orient="vertical", command=tree.yview); tree.configure(yscroll=vsb.set); vsb.pack(side="right", fill="y")
    tree.pack(fill="both", expand=True)

    def load_history():
        for it in tree.get_children(): tree.delete(it)
        rows = fetch_user_predictions(username)
        for r in rows:
            tree.insert("", tk.END, values=(r[0], r[1], r[3], r[5]))
    load_history()

    def on_history_double(ev):
        sel = tree.selection()
        if not sel: return
        pid = tree.item(sel[0])["values"][0]
        cur.execute("SELECT disease,input_json,result,advice,timestamp FROM predictions WHERE id=?", (pid,))
        rr = cur.fetchone()
        if rr:
            dis, inp, res, adv, ts = rr[0], rr[1], rr[2], rr[3], rr[4]
            inp = json.loads(inp)
            txt = f"Disease: {dis}\nResult: {res}\nTime: {ts}\n\nInputs:\n"
            for k,v in inp.items(): txt += f" - {k}: {v}\n"
            txt += f"\nAdvice:\n{adv}"
            messagebox.showinfo("Record details", txt)
    tree.bind("<Double-1>", on_history_double)

    # prediction worker (thread)
    def do_predict_thread():
        disease = disease_cb.get()
        model = MODELS.get(disease)
        if model is None:
            messagebox.showerror("Model missing", f"No model for {disease}. Use Admin -> Upload model.")
            return
        # collect inputs
        try:
            values=[]; inp={}
            for label, (entry, typ) in field_entries.items():
                v = entry.get().strip()
                if v == "": raise ValueError(f"{label} required")
                val = int(v) if typ=="int" else float(v)
                inp[label]=val; values.append(val)
        except Exception as e:
            messagebox.showerror("Input error", str(e)); return
        arr = np.array([values])
        try:
            pred = int(model.predict(arr)[0])
        except Exception as e:
            messagebox.showerror("Prediction error", str(e)); return
        result_text = "High Risk" if pred==1 else "Low Risk"
        advice_list = ADVICE.get(disease, ["Consult doctor."]) if pred==1 else ["Maintain healthy lifestyle."]
        advice_text = "\n".join(advice_list)
        # save
        save_prediction_db(username, disease, inp, result_text, advice_text)
        # update UI (must be on main thread)
        def finish():
            res_var.set(f"{result_text} - {disease}")
            if pred==1: res_label.configure(text_color="#e74c3c")
            else: res_label.configure(text_color="#27ae60")
            advice_box.delete("1.0","end"); advice_box.insert("1.0", advice_text)
            # notify
            msg = f"{disease} check result: {result_text}. Advice: {advice_list[0]}"
            if notify_email.get() and email:
                send_email(email, f"{disease} result", msg)
            if notify_sms.get() and phone:
                send_sms(phone, msg)
            load_history()
            messagebox.showinfo("Saved", f"Prediction saved: {result_text}")
        win.after(10, finish)

    btn_predict = ctk.CTkButton(left, text="Predict", command=lambda: threading.Thread(target=do_predict_thread, daemon=True).start())
    btn_predict.pack(pady=6)
    ctk.CTkButton(
    left,
    text="Chat with Doctor",
    fg_color="#2980b9",
    command=lambda: open_patient_chat(username)
).pack(pady=6)
    ctk.CTkButton(
    left,
    text="AI Health Chatbot",
    fg_color="#27ae60",
    command=lambda: open_ai_chatbot()
).pack(pady=6)


    ctk.CTkButton(left, text="Logout", fg_color="#c0392b", command=lambda: (fade_out_and_destroy(win), open_login_window())).pack(pady=4)

    # fade in
    win.attributes("-alpha",0.0); win.update(); fade_in(win)

# ---------- Doctor window ----------
def open_doctor_window():
    win = ctk.CTkToplevel(ROOT); win.title("Doctor Dashboard"); win.geometry("1000x700"); win.minsize(900,650)
    frame = ctk.CTkFrame(win); frame.pack(fill="both", expand=True, padx=8, pady=8)
    filter_frame = ctk.CTkFrame(frame); filter_frame.pack(fill="x", pady=6)
    tk.Label(filter_frame, text="Patient:").grid(row=0,column=0,padx=6); user_filter = ctk.CTkEntry(filter_frame); user_filter.grid(row=0,column=1,padx=6)
    tk.Label(filter_frame, text="Disease:").grid(row=0,column=2,padx=6); disease_filter = ttk.Combobox(filter_frame, values=["","Diabetes","Heart","Cancer"], state="readonly"); disease_filter.grid(row=0,column=3,padx=6)
    tk.Label(filter_frame, text="From:").grid(row=0,column=4,padx=6); from_f = ctk.CTkEntry(filter_frame,width=12); from_f.grid(row=0,column=5,padx=6)
    tk.Label(filter_frame, text="To:").grid(row=0,column=6,padx=6); to_f = ctk.CTkEntry(filter_frame,width=12); to_f.grid(row=0,column=7,padx=6)
    ctk.CTkButton(filter_frame, text="Apply", command=lambda: load_table(user_filter.get().strip() or None, disease_filter.get() or None, from_f.get().strip() or None, to_f.get().strip() or None)).grid(row=0,column=8,padx=6)
    ctk.CTkButton(filter_frame, text="Show All", fg_color="#010607", command=lambda: load_table()).grid(row=0,column=9,padx=6)

    table_frame = ctk.CTkFrame(frame); table_frame.pack(fill="both", expand=True, pady=8)
    cols = ("ID","User","Disease","Result","Time")
    tree = ttk.Treeview(table_frame, columns=cols, show="headings")
    for ccol in cols: tree.heading(ccol, text=ccol); tree.column(ccol, anchor="center", width=180)
    tree.pack(side="left", fill="both", expand=True)
    vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview); tree.configure(yscroll=vsb.set); vsb.pack(side="right", fill="y")
    ctrl_frame = ctk.CTkFrame(frame); ctrl_frame.pack(fill="x", pady=6)
    ctk.CTkButton(ctrl_frame, text="Show Risk Chart", command=lambda: show_chart(user_filter.get().strip() or None, disease_filter.get() or None, from_f.get().strip() or None, to_f.get().strip() or None)).pack(side="left", padx=6)
    ctk.CTkButton(ctrl_frame, text="Export CSV", fg_color="#16a085", command=lambda: export_csv(user_filter.get().strip() or None, disease_filter.get() or None, from_f.get().strip() or None, to_f.get().strip() or None)).pack(side="left", padx=6)
    ctk.CTkButton(ctrl_frame, text="Back (Logout)", fg_color="#c0392b", command=lambda: (fade_out_and_destroy(win), open_login_window())).pack(side="right", padx=6)
    ctk.CTkButton(
    ctrl_frame,
    text="Chat with Patient",
    fg_color="#8e44ad",
    command=open_doctor_chat
).pack(side="left", padx=6)


    def load_table(u=None,d=None,f=None,t=None):
        for it in tree.get_children(): tree.delete(it)
        rows = fetch_all_predictions(u,d,f,t)
        for r in rows:
            tree.insert("", tk.END, values=(r[0], r[1], r[2], r[4], r[6]))

    def show_chart(u=None,d=None,f=None,t=None):
        rows = fetch_all_predictions(u,d,f,t)
        if not rows: messagebox.showinfo("No data","No records"); return
        cnt={}
        for r in rows: cnt[r[4]] = cnt.get(r[4],0)+1
        labels=list(cnt.keys()); sizes=list(cnt.values())
        plt.figure(figsize=(5,5)); plt.pie(sizes, labels=labels, autopct="%1.1f%%"); plt.title("Risk Distribution"); plt.show()

    def export_csv(u=None,d=None,f=None,t=None):
        rows = fetch_all_predictions(u,d,f,t)
        if not rows: messagebox.showinfo("No data","No records"); return
        fp = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")], initialfile="predictions_export.csv")
        if not fp: return
        import csv
        with open(fp,"w",newline="",encoding="utf-8") as fw:
            w = csv.writer(fw); w.writerow(["ID","User","Disease","InputJSON","Result","Advice","Timestamp"]); w.writerows(rows)
        messagebox.showinfo("Exported", f"Saved CSV to {fp}")

    def on_double(ev):
        sel = tree.selection(); 
        if not sel: return
        pid = tree.item(sel[0])["values"][0]
        cur.execute("SELECT username,disease,input_json,result,advice,timestamp FROM predictions WHERE id=?", (pid,))
        rr = cur.fetchone()
        if rr:
            un,dis,inp,res,adv,ts = rr
            inp = json.loads(inp)
            ttxt = f"User: {un}\nDisease: {dis}\nResult: {res}\nTime: {ts}\n\nInputs:\n"
            for k,v in inp.items(): ttxt += f" - {k}: {v}\n"
            ttxt += f"\nAdvice:\n{adv}"
            messagebox.showinfo("Details", ttxt)
    tree.bind("<Double-1>", on_double)
    load_table()
    win.attributes("-alpha",0.0); win.update(); fade_in(win)

# ---------- Admin: Upload model mapping ----------
def open_admin_upload():
    win = ctk.CTkToplevel(ROOT); win.title("Admin - Upload Model"); win.geometry("600x260"); win.resizable(False,False)
    ctk.CTkLabel(win, text="Upload .pkl Model and map to disease", font=("Segoe UI", 14, "bold")).pack(pady=8)
    frame = ctk.CTkFrame(win); frame.pack(padx=12, pady=6, fill="both", expand=True)
    disease_var = tk.StringVar(value="Diabetes")
    ttk.Combobox(frame, values=["Diabetes","Heart","Cancer"], textvariable=disease_var, state="readonly").pack(pady=8)
    lbl_file = ctk.CTkLabel(frame, text="No file selected"); lbl_file.pack(pady=6)
    def choose_file():
        fp = filedialog.askopenfilename(filetypes=[("Pickle","*.pkl")])
        if fp: lbl_file.configure(text=os.path.basename(fp)); frame.selected = fp
    def upload():
        fp = getattr(frame, "selected", None)
        if not fp: messagebox.showerror("Error","Select a file"); return
        dest = disease_var.get().lower() + ".pkl"
        try:
            with open(fp,"rb") as fr, open(dest,"wb") as fw:
                fw.write(fr.read())
            MODELS[disease_var.get()] = load_model_file(dest)
            messagebox.showinfo("Uploaded", f"Uploaded and mapped to {disease_var.get()}")
        except Exception as e:
            messagebox.showerror("Upload failed", str(e))
    ctk.CTkButton(frame, text="Choose File", command=choose_file).pack(pady=6)
    ctk.CTkButton(frame, text="Upload & Map", command=upload).pack(pady=6)
    ctk.CTkButton(frame, text="Close", fg_color="#031213", command=lambda: fade_out_and_destroy(win)).pack(pady=6)
    win.attributes("-alpha",0.0); win.update(); fade_in(win)

# ---------- Main launcher UI ----------
def open_admin_panel_from_root():
    open_admin_upload()

def setup_root_ui():
    ROOT.configure(fg_color="#0D2852")
    title = ctk.CTkLabel(ROOT, text="AI Smart Healthcare", font=("Segoe UI", 22, "bold")); title.pack(pady=12)
    sub = ctk.CTkLabel(ROOT, text="Launcher - open Login / Admin panels (each opens as separate animated window)", font=("Segoe UI", 11)); sub.pack(pady=6)
    btn_frame = ctk.CTkFrame(ROOT); btn_frame.pack(pady=18)
    ctk.CTkButton(btn_frame, text="Open Login Window", command=open_login_window, width=200).grid(row=0,column=0,padx=12, pady=6)
    ctk.CTkButton(btn_frame, text="Open Admin Upload", fg_color="#16a085", command=open_admin_upload, width=200).grid(row=0,column=1, padx=12, pady=6)
    lbl = ctk.CTkLabel(ROOT, text="(Default doctor: doctor / doc123)", text_color="#7f8c8d"); lbl.pack(pady=8)

# ---------- Start app ----------
if __name__ == "__main__":
    setup_root_ui()
    ROOT.mainloop()
    conn.close()


