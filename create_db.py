from app import app
from models import db

with app.app_context():
    # Drop all old tables
    db.drop_all()
    print("❌ Old tables dropped.")

    # Create new tables as per current models
    db.create_all()
    print("✅ New tables created.")
