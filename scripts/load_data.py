import pandas as pd
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# 1. Establish database connection
engine = create_engine('mysql+pymysql://root:root123@localhost:3306/medireach')

# 2. Load Kaggle CSV dataset
print("Loading CSV...")
df = pd.read_csv('data/raw/KaggleV2-May-2016.csv')
print(f"Loaded {len(df)} rows")

# 3. Handle raw typos and map to clean snake_case names
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

# 4. Clean data types to match your relational schema rules
df['no_show'] = (df['no_show'] == 'Yes').astype(int)
df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
df['appointment_day'] = pd.to_datetime(df['appointment_day'])

# 5. Extract unique patients (Table 1)
patients_df = df[['patient_id', 'gender', 'age', 'neighbourhood', 
                  'scholarship', 'hypertension', 'diabetes', 
                  'alcoholism', 'handicap']].drop_duplicates(subset=['patient_id'])
print(f"Unique patients parsed: {len(patients_df)}")

# 6. Extract appointments (Table 2)
appointments_df = df[['appointment_id', 'patient_id', 'scheduled_day', 
                      'appointment_day', 'sms_received', 'no_show']]
print(f"Appointments parsed: {len(appointments_df)}")

# 7. Write clean structured frames to your MySQL database
print("\nLoading patients table...")
patients_df.to_sql('patients', engine, if_exists='replace', index=False)

print("Loading appointments table...")
appointments_df.to_sql('appointments', engine, if_exists='replace', index=False)

print("\n DATA LOADED SUCCESSFULLY!")

# 8. Run terminal schema validation checks
with engine.connect() as conn:
    res_p = conn.execute(text("SELECT COUNT(*) FROM patients"))
    print(f"Final Patients DB Count: {res_p.fetchone()[0]}")
    
    res_a = conn.execute(text("SELECT COUNT(*) FROM appointments"))
    print(f"Final Appointments DB Count: {res_a.fetchone()[0]}")
