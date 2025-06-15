from flask import Flask, render_template, request, redirect, url_for, session, flash
from models import db, Admin, Patient, SleepData
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sleep_apnea.db'
app.config['SECRET_KEY'] = 'secret123'
app.config['UPLOAD_FOLDER'] = 'uploads'

db.init_app(app)

with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login/<role>', methods=['GET', 'POST'])
def login(role):
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = Admin.query.filter_by(email=email).first() if role == 'admin' else Patient.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['role'] = role
            return redirect(url_for(f"{role}_dashboard"))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html', role=role)


@app.route('/register/<role>', methods=['GET', 'POST'])
def register(role):
    if request.method == 'POST':
        name = request.form['name'].strip()
        email = request.form['email'].strip()
        phone = request.form['phone'].strip()
        password = generate_password_hash(request.form['password'])

        email_regex = r'^[a-z][a-zA-Z0-9._%+-]*@gmail\.com$'
        phone_regex = r'^\+91[0-9]{10}$'

        if not re.fullmatch(email_regex, email):
            flash('❌ Invalid email format', 'danger')
            return redirect(request.url)
        if not re.fullmatch(phone_regex, phone):
            flash('❌ Invalid phone format', 'danger')
            return redirect(request.url)

        if role == 'admin':
            user = Admin(name=name, email=email, phone=phone, password=password)
        else:
            user = Patient(
                name=name,
                email=email,
                phone=phone,
                password=password,
                age=int(request.form['age']),
                weight=float(request.form['weight']),
                sex=request.form['sex'],
                marital_status=request.form['marital_status']
            )

        db.session.add(user)
        db.session.commit()
        flash('✅ Registration successful!', 'success')
        return redirect(url_for('login', role=role))

    return render_template('register.html', role=role)


@app.route('/forgot/<role>', methods=['GET', 'POST'])
def forgot_password(role):
    if request.method == 'POST':
        email = request.form['email']
        flash(f"If {email} exists, a reset link will be sent (demo only)", 'info')
        return redirect(url_for('login', role=role))
    return render_template('forgot.html', role=role)


@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('index'))

    prediction_result = None

    if request.method == 'POST':
        try:
            patient_id = request.form['patient_id'].strip()
            patient_name = request.form['patient_name'].strip()
            file = request.files['csv_file']

            if file:
                # Save uploaded file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                # Read CSV with correct headers
                df = pd.read_csv(file_path)  # ✅ Let pandas infer headers

                expected_cols = {'start_time', 'end_time', 'heart_rate', 'spo2'}
                if not expected_cols.issubset(set(df.columns)):
                    flash("❌ CSV must have columns: tart_time, end_time, heart_rate, spo2", "danger")
                    return redirect(url_for('admin_dashboard'))

                # Predict apnea using only HR + SpO2
                model = joblib.load("naive_bayes_apnea_model.pkl")
                scaler = joblib.load("naive_bayes_scaler.pkl")

                features = df[['heart_rate', 'spo2']]
                scaled_features = scaler.transform(features)
                predictions = model.predict(scaled_features)

                df['apnea_predicted'] = predictions

                # Save prediction result
                report_name = f"patient_{patient_id}_{patient_name}_report.csv"
                df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], report_name), index=False)

                apnea_count = int(df['apnea_predicted'].sum())
                total = len(df)

                prediction_result = (
                    f"😷 Apnea Detected in {apnea_count}/{total} intervals"
                    if apnea_count else f"✅ No Apnea Detected (Total intervals: {total})"
                )

        except Exception as e:
            flash(f"❌ Error processing file: {e}", "danger")
            return redirect(url_for('admin_dashboard'))

    return render_template("admin_dashboard.html", prediction_result=prediction_result)


@app.route('/patient_dashboard')
def patient_dashboard():
    if session.get('role') != 'patient':
        return redirect(url_for('index'))

    patient = Patient.query.get(session['user_id'])

    # ✅ Check if report data exists for the patient
    has_report = SleepData.query.filter_by(patient_id=patient.id).count() > 0

    return render_template(
        'patient_dashboard.html',
        patient_name=patient.name,
        has_report=has_report
    )



@app.route('/report')
def report():
    # Get the correct patient ID
    if session.get('role') == 'patient':
        patient_id = session['user_id']
    else:
        patient_id = 1  # default if admin views manually

    patient = Patient.query.get(patient_id)
    data = SleepData.query.filter_by(patient_id=patient_id).all()

    if not data:
        flash("No report exists for this patient.", "warning")
        return redirect(url_for('admin_dashboard' if session['role'] == 'admin' else 'patient_dashboard'))

    # Convert DB rows to DataFrame
    df = pd.DataFrame([{
        'date': d.date,
        'start_time': d.start_time,
        'end_time': d.end_time,
        'heart_rate': d.heart_rate,
        'spo2': d.spo2
    } for d in data])

    # ✅ Predict apnea
    try:
        model = joblib.load("naive_bayes_apnea_model.pkl")
        scaler = joblib.load("naive_bayes_scaler.pkl")
        X = df[['heart_rate', 'spo2']]
        X_scaled = scaler.transform(X)
        df['apnea_predicted'] = model.predict(X_scaled)
    except Exception as e:
        print("⚠️ Apnea prediction error:", e)
        df['apnea_predicted'] = 0  # fallback if error

    # ✅ Apnea summary
    apnea_count = int(df['apnea_predicted'].sum())
    total = len(df)
    apnea_summary = (
        f"😷 Apnea Detected in {apnea_count}/{total} intervals."
        if apnea_count else f"✅ No Apnea Detected (Total intervals: {total})."
    )

    # Labels for charts
    df['datetime'] = df['date'] + ' ' + df['start_time']
    x_labels = df['datetime'].tolist()
    heart_rate_data = df['heart_rate'].tolist()
    spo2_data = df['spo2'].tolist()

    # Smooth data
    window = len(df) if len(df) % 2 == 1 else len(df) - 1
    if window >= 5:
        hr_data = savgol_filter(df['heart_rate'], window_length=window, polyorder=2)
        spo2_data_smoothed = savgol_filter(df['spo2'], window_length=window, polyorder=2)
    else:
        hr_data = df['heart_rate']
        spo2_data_smoothed = df['spo2']

    x = np.arange(len(df))
    display_labels = [label if i % max(1, len(x_labels)//10) == 0 else "" for i, label in enumerate(x_labels)]

    # Save graphs
    plt.figure(figsize=(10, 3))
    plt.plot(x, hr_data, color='lime', linewidth=2)
    plt.title("HEART RATE MONITOR (bpm)", color='lime')
    plt.ylabel("bpm", color='white')
    plt.xticks(ticks=x, labels=display_labels, rotation=45, color='white')
    plt.yticks(color='white')
    plt.gca().set_facecolor('black')
    plt.tight_layout()
    plt.savefig("static/heart_rate_waveform.png", facecolor='black')
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(x, spo2_data_smoothed, color='cyan', linewidth=2)
    plt.title("SpO₂ MONITOR (%)", color='cyan')
    plt.ylabel("%", color='white')
    plt.xticks(ticks=x, labels=display_labels, rotation=45, color='white')
    plt.yticks(color='white')
    plt.gca().set_facecolor('black')
    plt.tight_layout()
    plt.savefig("static/spo2_waveform.png", facecolor='black')
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(x, hr_data, label='Heart Rate', color='lime')
    plt.plot(x, spo2_data_smoothed, label='SpO₂', color='cyan')
    plt.title("REAL-TIME VITAL MONITOR", color='yellow')
    plt.ylabel("Value", color='white')
    plt.xticks(ticks=x, labels=display_labels, rotation=45, color='white')
    plt.yticks(color='white')
    plt.legend(frameon=False)
    plt.gca().set_facecolor('black')
    plt.tight_layout()
    plt.savefig("static/combined_waveform.png", facecolor='black')
    plt.close()

    # Averages
    avg_heart_rate = round(df['heart_rate'].mean(), 2)
    avg_spo2 = round(df['spo2'].mean(), 2)

    return render_template(
        'report.html',
        patient=patient,
        avg_heart_rate=avg_heart_rate,
        avg_spo2=avg_spo2,
        labels=x_labels,
        heart_rate=heart_rate_data,
        spo2=spo2_data,
        apnea_summary=apnea_summary  # ✅ Added!
    )


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
