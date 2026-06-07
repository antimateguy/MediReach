import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import BigInteger, Boolean

# Database connection
engine = create_engine('mysql+pymysql://root:root123@localhost:3306/medireach')

print("=" * 50)
print("MediReach - Data Loader")
print("=" * 50)

# Load CSV
print("\n[1/5] Loading CSV file...")
df = pd.read_csv('data/raw/KaggleV2-May-2016.csv')
print(f"     ✓ Loaded {len(df):,} rows")

# Rename columns first for clarity
df.rename(columns={
    'PatientId': 'patient_id',
    'AppointmentID': 'appointment_id',
    'Gender': 'gender',
    'ScheduledDay': 'scheduled_day',
    'AppointmentDay': 'appointment_day',
    'Age': 'age',
    'Neighbourhood': 'neighbourhood',
    'Scholarship': 'scholarship',
    'Hipertension': 'hypertension',
    'Diabetes': 'diabetes',
    'Alcoholism': 'alcoholism',
    'Handcap': 'handicap',
    'SMS_received': 'sms_received',
    'No-show': 'no_show'
}, inplace=True)

# Create patients table (unique patients)
print("\n[2/5] Creating patients data (deduplicated)...")
patients_df = df[[
    'patient_id', 'gender', 'age', 'neighbourhood',
    'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'handicap'
]].drop_duplicates(subset=['patient_id'])

# Convert to boolean
bool_cols = ['scholarship', 'hypertension', 'diabetes', 'alcoholism', 'handicap']
for col in bool_cols:
    patients_df[col] = patients_df[col].astype(bool)

print(f"     ✓ {len(patients_df):,} unique patients")

# Create appointments table
print("\n[3/5] Creating appointments data...")
appointments_df = df[[
    'appointment_id', 'patient_id', 'scheduled_day', 'appointment_day',
    'sms_received', 'no_show'
]].copy()

# Convert to boolean
appointments_df['sms_received'] = appointments_df['sms_received'].astype(bool)
appointments_df['no_show'] = (appointments_df['no_show'] == 'Yes')

# Convert dates
appointments_df['scheduled_day'] = pd.to_datetime(appointments_df['scheduled_day'])
appointments_df['appointment_day'] = pd.to_datetime(appointments_df['appointment_day'])

print(f"     ✓ {len(appointments_df):,} appointments")

# Load to MySQL (tables are empty, so no drop conflict)
print("\n[4/5] Loading patients to MySQL...")
patients_df.to_sql('patients', engine, if_exists='append', index=False,
                   dtype={col: BigInteger for col in ['patient_id']})
print("     ✓ Patients table loaded")

print("\n[5/5] Loading appointments to MySQL...")
appointments_df.to_sql('appointments', engine, if_exists='append', index=False,
                       dtype={col: BigInteger for col in ['appointment_id', 'patient_id']})
print("     ✓ Appointments table loaded")

# Verify
print("\n" + "=" * 50)
print("VERIFICATION")
print("=" * 50)

with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM patients"))
    patients_count = result.fetchone()[0]
    print(f"Patients count:  {patients_count:,}")
    
    result = conn.execute(text("SELECT COUNT(*) FROM appointments"))
    appointments_count = result.fetchone()[0]
    print(f"Appointments count: {appointments_count:,}")
    
    # Check relationship
    result = conn.execute(text("""
        SELECT COUNT(*) 
        FROM appointments a
        LEFT JOIN patients p ON a.patient_id = p.patient_id
        WHERE p.patient_id IS NULL
    """))
    orphaned = result.fetchone()[0]
    print(f"Orphaned appointments (no patient): {orphaned}")
    
    # Sample data
    result = conn.execute(text("""
        SELECT p.patient_id, p.gender, p.age, COUNT(a.appointment_id) as appt_count
        FROM patients p
        JOIN appointments a ON p.patient_id = a.patient_id
        GROUP BY p.patient_id
        LIMIT 5
    """))
    print("\nSample patients with appointment counts:")
    for row in result:
        print(f"  Patient {row[0]}: {row[1]}, age {row[2]} → {row[3]} appointments")

print("\n✅ DATA LOAD COMPLETE!")

# Create outreach_queue table (empty for now)
print("\n[Bonus] Creating outreach_queue table...")
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS outreach_queue (
            queue_id INT AUTO_INCREMENT PRIMARY KEY,
            patient_id BIGINT,
            appointment_id BIGINT,
            risk_score DECIMAL(5,2),
            priority VARCHAR(20),
            recommended_action VARCHAR(50),
            generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY (appointment_id) REFERENCES appointments(appointment_id)
        )
    """))
    conn.commit()
print("     ✓ outreach_queue table ready")

