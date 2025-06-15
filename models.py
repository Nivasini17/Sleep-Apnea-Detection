from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    weight = db.Column(db.Float)
    sex = db.Column(db.String(10))
    marital_status = db.Column(db.String(20))

class SleepData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'))
    date = db.Column(db.String(50))
    start_time = db.Column(db.String(10))
    end_time = db.Column(db.String(10))
    heart_rate = db.Column(db.Integer)
    spo2 = db.Column(db.Integer)
    apnea = db.Column(db.Integer)  # ✅ New column for prediction

