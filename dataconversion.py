import os
import scipy.io
import pandas as pd
import numpy as np

# === Folder Paths ===
BASE_FOLDER = 'HuGCDN2014-OXI'
RR_FOLDER = os.path.join(BASE_FOLDER, 'RR')
SAT_FOLDER = os.path.join(BASE_FOLDER, 'SAT')
LABELS_FOLDER = os.path.join(BASE_FOLDER, 'LABELS')

# === .mat File Keys ===
RR_KEY = 'RR_notch_abs_pr_ada'
SAT_KEY = 'SAT'
LABEL_KEY = 'salida_man_1m'

# === Safe Flattening Loader ===
def load_mat_array(path, key):
    try:
        mat = scipy.io.loadmat(path)
        raw = mat.get(key)

        if raw is None:
            return np.array([])

        # Force flatten using pure Python + NumPy
        flattened = []

        def extract_numbers(obj):
            if isinstance(obj, (list, tuple, np.ndarray)):
                for item in obj:
                    extract_numbers(item)
            else:
                try:
                    flattened.append(float(obj))
                except:
                    pass

        extract_numbers(raw)
        return np.array(flattened, dtype=np.float64)

    except Exception as e:
        print(f"❌ Error loading {key} from {path}: {e}")
        return np.array([])


# === Matched Files (Present in All 3 Folders) ===
rr_files = {f for f in os.listdir(RR_FOLDER) if f.endswith('.mat')}
sat_files = {f for f in os.listdir(SAT_FOLDER) if f.endswith('.mat')}
label_files = {f for f in os.listdir(LABELS_FOLDER) if f.endswith('.mat')}
common_files = rr_files & sat_files & label_files

all_data = []

# === Main Extraction Loop ===
for file in sorted(common_files):
    rr_path = os.path.join(RR_FOLDER, file)
    sat_path = os.path.join(SAT_FOLDER, file)
    label_path = os.path.join(LABELS_FOLDER, file)

    rr = load_mat_array(rr_path, RR_KEY)
    sat = load_mat_array(sat_path, SAT_KEY)
    labels = load_mat_array(label_path, LABEL_KEY)

    print(f"🧐 {file} → RR: {len(rr)}, SAT: {len(sat)}, LABELS: {len(labels)}")

    if len(rr) == 0 or len(sat) == 0 or len(labels) == 0:
        print(f"⚠️ Skipping {file} — one or more arrays empty\n")
        continue

    min_len = min(len(rr), len(sat), len(labels))
    rr, sat, labels = rr[:min_len], sat[:min_len], labels[:min_len]

    # Convert RR intervals to heart rate in bpm
    heart_rate = np.clip(60000 / rr, 40, 180)

    df = pd.DataFrame({
        'heart_rate': heart_rate,
        'spo2': sat,
        'apnea': labels
    })

    all_data.append(df)
    print(f"✅ Processed {file} — {len(df)} rows\n")

# === Save CSV ===
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv('apnea_hr_spo2_dataset.csv', index=False)
    print("🎉 Saved: apnea_hr_spo2_dataset.csv")
else:
    print("❌ No valid data saved.")
