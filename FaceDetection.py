import cv2
import os
import numpy as np
from tkinter import *
from tkinter import messagebox, filedialog, simpledialog
from PIL import Image
import tkinter.font as tkFont
from datetime import datetime, date
import csv

# ============================================================
# Single-file Face Recognition Attendance App
# - Admin-only user registration
# - Attendance summary on top of login page
# - Login with UserID + Password
# - Face capture / Train / Recognize
# IMPORTANT: install opencv-contrib-python (for cv2.face) and pillow
# pip install opencv-contrib-python pillow
# ============================================================

# -----------------------
# Helpers & Directories
# -----------------------

def ensure_dirs():
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("trainer", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def get_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    return cap


# -----------------------
# User login & registration (CSV)
# -----------------------
# File: logs/users_login.csv -> columns: UserID,Password,Name,NumericID
# NOTE: Passwords are stored in plain text in this demo. For production, hash them.

USERS_FILE = "logs/users_login.csv"
ADMIN_USER = "admin"
ADMIN_PASS = "1234"


def init_users_file():
    ensure_dirs()
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["UserID", "Password", "Name", "NumericID"])


def save_login_user(user_id, password, name):
    init_users_file()
    numeric_id = abs(hash(user_id)) % 10000
    # Avoid duplicate UserID
    existing = read_all_users()
    if user_id in existing:
        return False, "UserID already exists"
    with open(USERS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([user_id, password, name, numeric_id])
    return True, numeric_id


def read_all_users():
    users = {}
    if not os.path.exists(USERS_FILE):
        return users
    with open(USERS_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users[row["UserID"]] = {
                "password": row["Password"],
                "name": row["Name"],
                "numeric_id": int(row["NumericID"]) if row.get("NumericID") else None,
            }
    return users


def authenticate_user(user_id, password):
    if user_id == ADMIN_USER and password == ADMIN_PASS:
        return True, "Admin"
    users = read_all_users()
    if user_id in users and users[user_id]["password"] == password:
        return True, users[user_id]["name"]
    return False, None


def get_user_name_by_numeric(numeric_id):
    users = read_all_users()
    for uid, meta in users.items():
        if meta.get("numeric_id") == numeric_id:
            return meta.get("name") or uid
    return f"{numeric_id}"


# -----------------------
# Attendance logging
# -----------------------
ATT_FILE = "logs/attendance.csv"


def init_attendance_file():
    ensure_dirs()
    if not os.path.exists(ATT_FILE):
        with open(ATT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])


def mark_attendance(name):
    init_attendance_file()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # prevent duplicate for same day + name
    with open(ATT_FILE, "r", newline="") as f:
        rows = f.read().splitlines()
        if any(name in r and date_str in r for r in rows):
            return False

    with open(ATT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])
    return True


def get_attendance_summary():
    init_attendance_file()
    today = date.today().strftime("%Y-%m-%d")
    total_today = 0
    last_entries = []
    if not os.path.exists(ATT_FILE):
        return total_today, last_entries
    with open(ATT_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Date"] == today:
                total_today += 1
            last_entries.append((row["Name"], row["Date"], row["Time"]))
    # return count and last 5 entries (most recent at end)
    return total_today, last_entries[-5:]


# -----------------------
# User mapping for recognizer
# -----------------------

MAPPING_FILE = "logs/users.csv"  # simpler mapping created when registering or capturing


def save_user_mapping(user_id, numeric_id):
    ensure_dirs()
    if not os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["UserID", "NumericID"])
    with open(MAPPING_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([user_id, numeric_id])


def get_user_name(numeric_id):
    # prefer login file lookup
    users = read_all_users()
    for uid, meta in users.items():
        if meta.get("numeric_id") == numeric_id:
            return meta.get("name") or uid
    # fallback to mapping file
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(numeric_id) == row["NumericID"]:
                    return row["UserID"]
    return f"User_{numeric_id}"


# -----------------------
# Face capture / training / recognition
# -----------------------


def capture_faces(root, current_user_id=None):
    ensure_dirs()
    # If opened from logged-in user, use that ID; else ask
    if current_user_id:
        user_id = current_user_id
    else:
        user_id = simpledialog.askstring("Capture Face", "Enter User ID or Name:")

    if not user_id:
        messagebox.showwarning("Warning", "User ID cannot be empty.")
        return

    numeric_id = abs(hash(user_id)) % 100
    # Save mapping both in login file (if exists) and mapping file
    users = read_all_users()
    if user_id in users:
        # if exists, ensure numeric id consistent
        numeric_id = users[user_id]["numeric_id"]
    else:
        # not in login table — still save mapping for recognizer
        save_user_mapping(user_id, numeric_id)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = get_camera()

    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access camera.")
        return

    count = 0
    messagebox.showinfo("Instructions", "Press 'q' to stop capturing. Move your face to different angles.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y + h, x:x + w]
            cv2.imwrite(f"dataset/{user_id}_{count}.jpg", face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capturing Faces - Press 'q' to Stop", frame)
        if cv2.waitKey(3) & 0xFF == ord('q') or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Saved", f"Captured {count} faces for {user_id}")


def train_faces(root):
    ensure_dirs()
    # require opencv-contrib for cv2.face
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_samples = []
    ids = []

    image_paths = [os.path.join("dataset", f) for f in os.listdir("dataset") if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for path in image_paths:
        gray_img = Image.open(path).convert("L")
        img_np = np.array(gray_img, "uint8")
        id_str = os.path.split(path)[-1].split("_")[0]
        numeric_id = abs(hash(id_str)) % 10000
        # if user exists in login, use stored numeric id
        users = read_all_users()
        if id_str in users:
            numeric_id = users[id_str]["numeric_id"]

        face_samples.append(img_np)
        ids.append(numeric_id)

    if not face_samples:
        messagebox.showerror("Error", "No faces found. Capture faces first!")
        return

    recognizer.train(face_samples, np.array(ids))
    recognizer.save("trainer/trainer.yml")
    messagebox.showinfo("Success", "Model Trained Successfully!")


def recognize_from_webcam(root):
    ensure_dirs()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read("trainer/trainer.yml")
    except Exception:
        messagebox.showerror("Error", "Train the model first!")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = get_camera()

    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access camera.")
        return

    messagebox.showinfo("Info", "Press 'q' to close webcam window.")

    recorded = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            try:
                id_pred, conf = recognizer.predict(gray[y:y + h, x:x + w])
            except Exception:
                id_pred, conf = -1, 999
            name = get_user_name_by_numeric(id_pred)
            label = name if conf < 80 else "Unknown"

            if label != "Unknown" and label not in recorded:
                marked = mark_attendance(label)
                recorded.add(label)
                if marked:
                    messagebox.showinfo("Attendance", f"Attendance marked for {label}")

            cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Recognition - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------
# Admin-only registration
# -----------------------


def admin_register_prompt(parent):
    # Ask for admin password first
    pwd = simpledialog.askstring("Admin Authentication", "Enter admin password:", show='*', parent=parent)
    if pwd != ADMIN_PASS:
        messagebox.showerror("Error", "Incorrect admin password.")
        return

    # Now show registration form
    reg = Toplevel(parent)
    reg.title("Register New User (Admin)")
    reg.geometry("380x300")

    Label(reg, text="Register New User", font=("Helvetica", 16, "bold")).pack(pady=10)
    Label(reg, text="Full Name").pack()
    name_e = Entry(reg, width=30)
    name_e.pack()

    Label(reg, text="User ID (unique)").pack()
    uid_e = Entry(reg, width=30)
    uid_e.pack()

    Label(reg, text="Password").pack()
    pwd_e = Entry(reg, width=30, show='*')
    pwd_e.pack()

    def do_register():
        name = name_e.get().strip()
        uid = uid_e.get().strip()
        pwdval = pwd_e.get().strip()
        if not (name and uid and pwdval):
            messagebox.showwarning("Warning", "All fields are required.")
            return
        ok, res = save_login_user(uid, pwdval, name)
        if not ok:
            messagebox.showerror("Error", res)
        else:
            # also save mapping
            save_user_mapping(uid, res)
            messagebox.showinfo("Success", f"User {name} registered with numeric id {res}")
            reg.destroy()

    Button(reg, text="Register", command=do_register, bg="#4CAF50", fg="white").pack(pady=15)


# -----------------------
# GUI – Main App (after login)
# -----------------------


def open_main_app(login_window, logged_in_user):
    login_window.destroy()
    root = Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("1000x650")
    root.configure(bg="#f4a261")

    Label(root, text=f"Face Recognition Attendance System - Logged in as: {logged_in_user}",
          font=("Helvetica", 20, "bold"), bg="#f4a261").pack(pady=20)

    btn_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
    button_style = {"width": 25, "height": 2, "bd": 3, "relief": "raised",
                    "font": btn_font}

    frame = Frame(root, bg="#f4a261")
    frame.pack(pady=40)

    Button(frame, text="📸 Capture Face", bg="#90CAF9",
           command=lambda: capture_faces(root), **button_style).pack(pady=15)

    Button(frame, text="🧠 Train Model", bg="#A5D6A7",
           command=lambda: train_faces(root), **button_style).pack(pady=15)

    Button(frame, text="🎥 Recognize Face (Webcam)", bg="#FFAB91",
           command=lambda: recognize_from_webcam(root), **button_style).pack(pady=15)
    
    
    Button(frame, text="🖼 Recognize Face (Image)", bg="#FFF176", activebackground="#FDD835",
           **button_style, command=lambda: recognize_from_image(root)).pack(pady=15)

    Button(root, text="🚪 Exit", bg="red", fg="white",
           font=("Helvetica", 14, "bold"), width=10,
           command=root.destroy).pack(side=BOTTOM, pady=25)
    root.mainloop()


# small helper to recognize from a single image file

def recognize_from_image(root):
    ensure_dirs()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read("trainer/trainer.yml")
    except Exception:
        messagebox.showerror("Error", "Train the model first!")
        return

    path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not path:
        return

    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        id_pred, conf = recognizer.predict(gray[y:y + h, x:x + w])
        label = get_user_name_by_numeric(id_pred) if conf < 80 else "Unknown"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Recognized Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -----------------------
# Login UI (with attendance summary at top and admin-only register)
# -----------------------

def start_login():
    init_users_file()
    init_attendance_file()

    login_window = Tk()
    login_window.title("Face Recognition Attendance System - Login")
    login_window.geometry("420x420")
    login_window.configure(bg="#87CEEB")

    # Attendance summary displayed on top
    total_today, last_entries = get_attendance_summary()
    Label(login_window, text="Face Recognition Attendance System", font=("Helvetica", 18, "bold"), bg="#87CEEB").pack(pady=8)
    Label(login_window, text=f"Today's marked attendance: {total_today}", font=("Helvetica", 12), bg="#87CEEB").pack()

    if last_entries:
        txt = "Last entries:\n" + "\n".join([f"{n} @ {d} {t}" for (n, d, t) in last_entries])
    else:
        txt = "No attendance recorded yet."

    Label(login_window, text=txt, justify=LEFT, bg="#87CEEB").pack(pady=6)

    # Login fields
    Label(login_window, text="User ID:", bg="#87CEEB").pack(pady=(10, 0))
    user_entry = Entry(login_window, width=30)
    user_entry.pack()

    Label(login_window, text="Password:", bg="#87CEEB").pack(pady=(8, 0))
    pass_entry = Entry(login_window, show="*", width=30)
    pass_entry.pack()

    def do_login():
        uid = user_entry.get().strip()
        pwd = pass_entry.get().strip()
        ok, name = authenticate_user(uid, pwd)
        if ok:
            messagebox.showinfo("Success", f"Welcome {name}!")
            open_main_app(login_window, uid)
        else:
            messagebox.showerror("Error", "Invalid credentials")

    btn_frame = Frame(login_window, bg="#87CEEB")
    btn_frame.pack(pady=12)

    Button(btn_frame, text="Login", width=12, bg="#4CAF50", fg="white", command=do_login).grid(row=0, column=0, padx=6)
    Button(btn_frame, text="Register (Admin)", width=14, bg="#1976D2", fg="white", command=lambda: admin_register_prompt(login_window)).grid(row=0, column=1, padx=6)

    Label(login_window, text="\nDefault Admin -> User: admin  Pass: 1234", bg="#87CEEB").pack()

    login_window.mainloop()


# -----------------------
# Run Application
# -----------------------

if __name__ == "__main__":
    start_login()
