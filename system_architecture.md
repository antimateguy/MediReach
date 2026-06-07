# MediReach - Complete System Architecture

## Project Identity
Name: MediReach
Type: Healthcare Operations Intelligence Platform
Status: Day 2 Complete (40% done)



## Business Problem
Hospital outpatient clinics lose 15-30% revenue to appointment no-shows. Manual outreach is inefficient. Need data-driven prioritization.

## Solution
End-to-end decision support system: SQL Analytics → ML Prediction → Action Engine → Dashboard

## Database Schema (MySQL 8.0)

### Connection Details
Host: localhost
Port: 3306
Database: medireach
User: root
Password: root123



### Table 1: patients (62,299 rows)
| Column | Type | Description |
|--------|------|-------------|
| patient_id | double | Unique patient identifier (PK) |
| gender | text | M or F |
| age | bigint | Age in years |
| neighbourhood | text | Geographic region |
| scholarship | bigint | 0 or 1 (Bolsa Familia) |
| hypertension | bigint | 0 or 1 |
| diabetes | bigint | 0 or 1 |
| alcoholism | bigint | 0 or 1 |
| handicap | bigint | 0-4 (disability level) |

### Table 2: appointments (110,527 rows)
| Column | Type | Description |
|--------|------|-------------|
| appointment_id | bigint | Unique appointment ID (PK) |
| patient_id | double | Links to patients table (FK) |
| scheduled_day | timestamp | When appointment was booked |
| appointment_day | timestamp | Scheduled visit date |
| sms_received | bigint | 0 or 1 (reminder sent) |
| no_show | bigint | 0 or 1 (Target variable) |

### Table 3: outreach_queue (empty - to be populated)
| Column | Type | Description |
|--------|------|-------------|
| queue_id | int AUTO_INCREMENT | Primary key |
| patient_id | bigint | Patient to contact |
| appointment_id | bigint | Upcoming appointment |
| risk_score | decimal(5,2) | 0-100 probability |
| priority | varchar(20) | HIGH/MEDIUM/LOW |
| recommended_action | varchar(50) | Call/SMS/No action |
| generated_at | datetime | Timestamp of prediction |

### Analytics Views (5 total)
| View Name | Purpose | Window Functions |
|-----------|---------|------------------|
| vw_patient_risk_profile | Rank patients by no-show rate | ROW_NUMBER, AVG() OVER |
| vw_age_analysis | Segment by age group | None |
| vw_lead_time_analysis | Wait time impact | None |
| vw_neighbourhood_risk | Geographic risk ranking | RANK |
| vw_monthly_trend | MoM comparison | LAG |

## Module Architecture

### Module 1: Database Layer ✅ COMPLETE
**Files:** `scripts/load_data.py`, `sql/schema.sql`
**Status:** 3 tables created, 172,826 total rows loaded
**Time spent:** 2 hours

### Module 2: SQL Analytics Layer ✅ COMPLETE
**Files:** `sql/views.sql`, `docs/business_insights.md`
**Status:** 5 views created, key insights documented
**Key findings:**
- Lead time >7 days = 2.3x higher no-show risk
- Age 18-34 = 24% no-show (highest segment)
- Top 5 neighborhoods = 27-29% no-show rate
**Time spent:** 3 hours

### Module 3: Feature Engineering  Complete
**Files to create:** `notebooks/02_feature_engineering.ipynb`
**Tasks:**
- Create lead_days feature
- Create day_of_week feature
- Create patient_history feature
- Create age_bins one-hot encoding
- Create interaction features (scholarship × sms)
- Train/test split (80/20)
**Estimated time:** 2-3 hours

### Module 4: ML Layer  Complete
**Files to create:** `scripts/train_model.py`, `models/`
**Models:**
- Logistic Regression (primary - explainable)
- Random Forest (secondary - comparison)
**Metrics:** ROC-AUC, Precision, Recall
**Target:** AUC > 0.70 (not chasing 0.90)
**Estimated time:** 2-3 hours

### Module 5: Action Engine ⏳ PENDING
**Files to create:** `scripts/generate_queue.py`
**Logic:**
if risk_score > 0.80:
action = "Call patient"
priority = "HIGH"
elif risk_score > 0.60:
action = "SMS reminder"
priority = "MEDIUM"
else:
action = "No action"
priority = "LOW"


**Output:** Populate outreach_queue table
**Estimated time:** 1 hour

### Module 6: Power BI Dashboard ⏳ PENDING
**File:** `dashboard/medireach_dashboard.pbix`
**Pages:**
- Executive Overview (KPIs, trends)
- Risk Analysis (segments, neighborhoods)
- Operations Queue (daily outreach list)
**Estimated time:** 2-3 hours

### Module 7: Streamlit UI ⏳ PENDING
**File:** `app.py`
**Features:**
- Analytics tab (show insights from views)
- Risk Queue tab (show today's outreach)
- Business Impact tab (estimated savings)
**Deployment:** Streamlit Cloud (free)
**Estimated time:** 2-3 hours

## File Structure (Current)
MediReach/
├── .gitignore ✅
├── README.md ⏳ (needs content)
├── requirements.txt ⏳ (needs packages)
├── SYSTEM_ARCHITECTURE.md ✅ (this file)
│
├── data/
│ ├── raw/
│ │ └── KaggleV2-May-2016.csv ✅ (50MB source)
│ └── processed/ (empty for now)
│
├── docs/
│ └── business_insights.md ✅ (interview ready)
│
├── sql/
│ ├── schema.sql ✅ (table definitions)
│ └── views.sql ✅ (5 analytics views)
│
├── scripts/
│ └── load_data.py ✅ (CSV to MySQL)
│
├── notebooks/ (empty - Day 3)
├── models/ (empty - Day 4)
├── dashboard/ (empty - Day 5)
├── screenshots/ (empty - Day 7)
└── app.py (not created - Day 6)



## Tech Stack Details

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Database | MySQL | 8.0 | Data storage, SQL analytics |
| Python | CPython | 3.10 | ETL, ML, API |
| ML Library | scikit-learn | 1.3+ | Logistic Regression, RF |
| Data Processing | pandas | 2.0+ | Feature engineering |
| Visualization | Power BI | Desktop | Executive dashboard |
| UI | Streamlit | 1.28+ | Operations interface |
| Deployment | Streamlit Cloud | - | Live demo link |

## Data Flow Diagram
KaggleV2-May-2016.csv (110k rows)
↓
[load_data.py] - Split & Clean
↓
MySQL Database (3 tables)
↓
[SQL Views] - Analytics & KPIs
↓
[Feature Engineering] - Create ML features
↓
[Train Model] - Logistic Regression
↓
[Action Engine] - Risk scoring
↓
outreach_queue table
↓
├── Power BI Dashboard (Executive)
└── Streamlit App (Operations)



## Current Status (Real-time)

### ✅ COMPLETED (Day 1-2)
- [x] MySQL installation and configuration
- [x] Database `medireach` created
- [x] 3 tables schema designed
- [x] 172,826 rows loaded successfully
- [x] 5 analytics views with window functions
- [x] Business insights documented
- [x] Git repository initialized
- [x] .gitignore configured
- [x] Feature engineering notebook
- [x] ML model training (2 models)

### ⏳ PENDING (Day 3-7)

- [ ] Action engine script
- [ ] Power BI dashboard (3 pages)
- [ ] Streamlit application
- [ ] Requirements.txt generation
- [ ] README.md completion
- [ ] Streamlit Cloud deployment
- [ ] Screenshots for resume


### Key Metrics & Insights

## Data Quality Discoveries
- 35% of appointments had `scheduled_day > appointment_day` (data entry errors/walk-ins)
- Filtered to 71,959 valid future-prediction appointments
- No-show rate in cleaned data: 28.5% (matches Kaggle's "30% miss rate")
- Elderly patients (65+): 20.8% no-show (lowest risk)
- Young adults (18-34): 34.5% no-show (highest risk)

## FINAL DATA VALIDATION 

1. No infinite values: True
2. No null values: True
3. All features numeric: True

4. Feature ranges:
   - lead_days: 0 to 178
   - past_no_show_rate: 0.00 to 1.00
   - age: 0 to 115

5. Class balance: 28.52% no-show
6. No future data leakage: True


### Business Metrics
Overall No-Show Rate: 19.64%
Total Patients: 62,299
Total Appointments: 110,527
Time Period: April-June 2016


## ML Model Performance (Actual Results)

### Data Processing
- Original rows: 110,527
- After cleaning (valid future appointments): 71,959 rows
- Filtered out: 35% (scheduled_day > appointment_day - data entry errors)
- Final no-show rate: 28.52%

### Feature Engineering (26 total features)
**Top 5 predictors (Random Forest importance):**
1. past_no_show_rate (0.475) - Patient history
2. lead_x_past (0.254) - Interaction feature (lead_days × history)
3. lead_days (0.051) - Wait time
4. age_squared (0.047) - Non-linear age effect
5. age (0.045) - Demographics

**Engineered features added:**
- is_weekend, age_squared, lead_days_binned
- scholarship_young, chronic_count, handicap_squared
- lead_x_past, sms_young_effective

### Model Performance (Threshold = 0.6)
| Metric | Logistic Regression | Random Forest (200) |
|--------|---------------------|---------------------|
| ROC-AUC | **0.742** | 0.710 |
| Recall | **0.612** | 0.196 |
| Precision | 0.454 | **0.664** |
| F1-Score | **0.521** | 0.303 |

**Selected Model:** Logistic Regression (interpretable, better recall)

### Business Impact (Per 14,392 appointments)
- Actual no-shows: 4,104 (28.5%)
- Caught by model: 1,564 patients
- Missed: 2,540 patients
- False alarms: 1,188 patients (contacted unnecessarily)

## Success Criteria - Status Update

### Minimum Viable Product (MVP)
| Requirement | Status | Notes |
|-------------|--------|-------|
| MySQL with 3 tables and 5 views | ✅ COMPLETE | 172,826 rows loaded |
| Logistic regression model (AUC > 0.70) | ✅ COMPLETE | AUC = 0.742 |
| Action engine generating daily queue | ⏳ PENDING | Next module |
| Power BI dashboard with 3 pages | ⏳ PENDING | |
| Streamlit app deployed with live link | ⏳ PENDING | |

### Stretch Goals
| Goal | Status | Notes |
|------|--------|-------|
| Random Forest model for comparison | ✅ COMPLETE | Underperformed LR |
| Foreign key constraints added | ⏳ PENDING | Need to clean orphaned records |
| Historical trend analysis in dashboard | ⏳ PENDING | |
| SMS integration simulation | ⏳ PENDING | |

### Known Issues & Future Improvements
1. **Data quality:** 35% rows filtered out (scheduled_day > appointment_day) - document as real-world data issue
2. **Random Forest performance:** Needs more data (200k+ rows) to outperform LR
3. **Recall improvement:** Consider lowering threshold to 0.5 for higher catch rate
4. **Feature store:** Implement daily retraining pipeline


### Key Findings
✅ Model meets AUC target (>0.70)
✅ Patient history = strongest predictor
✅ Engineered interaction features add value
⚠️ Random Forest underperformed (needs more data)
⚠️ Recall at 0.6 threshold is 0.381 (acceptable for pilot)


### Critical Insights (For Interview)
1. **Lead time is strongest predictor** - 2.3x risk increase after 7 days
2. **Young adults highest risk** - 24% no-show vs 15.5% elderly
3. **Geographic clustering** - 5 neighborhoods 24-47% above average
4. **Patient history repeats** - Past no-shows predict future no-shows

### Operational Metrics (Actual)
**At threshold 0.6 (current deployment):**
- Catch rate: 38% of no-shows
- Precision: 57% of alerts are correct
- Daily outreach: ~83 patients (per 1,000 appointments)
- False alarms per day: ~25 patients

**Threshold tuning options for operations:**
| Threshold | Recall | Precision | False Alarms/1k |
|-----------|--------|-----------|-----------------|
| 0.5 | 61% | 45% | 210 |
| 0.6 (default) | 38% | 57% | 83 |
| 0.7 | 28% | 66% | 41 |

**Recommendation:** Start with 0.6, adjust based on outreach team capacity


## Interview Preparation

### 5 Key Talking Points
1. "I normalized the data into 3 tables showing 1-to-many relationship"
2. "Used window functions (ROW_NUMBER, RANK, LAG) for patient risk ranking"
3. "Discovered lead time is 2.3x stronger predictor than demographics"
4. "Chose Logistic Regression for stakeholder interpretability"
5. "Built action engine that generates daily outreach queue - not just prediction"





### Model Artifacts Saved
models/
├── logistic_regression.pkl # Production model (0.742 AUC)
├── random_forest_200.pkl # Comparison model
├── best_model.pkl # Alias to logistic_regression
├── scaler.pkl # StandardScaler for features
└── risk_threshold.txt # Current threshold (0.6)

metrics/
├── model_metrics.json # All metrics & thresholds
├── rf_feature_importance.csv # Top predictors
├── lr_coefficients.csv # Model coefficients (interpretability)
└── plots/
├── confusion_matrix_roc.png
└── precision_recall_curve.png


### Window Functions Examples (Know These)
```sql
-- ROW_NUMBER: Find each patient's most recent appointment
ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY appointment_day DESC)

-- RANK: Rank neighborhoods by no-show rate
RANK() OVER (ORDER BY no_show_rate DESC)

-- LAG: Compare current vs previous month
LAG(no_show_rate) OVER (ORDER BY month)
Agent Context Section
For AI Agents helping with this project:
Connection String:




mysql+pymysql://root:root123@localhost:3306/medireach
Key Variables:


TARGET_COLUMN = 'no_show'
PATIENT_ID_COLUMN = 'patient_id'
APPOINTMENT_ID_COLUMN = 'appointment_id'
Data Types to Handle:

patient_id: bigint (yes, not int - legacy from CSV)

age: bigint (has negative values - filter out)

handicap: 0-4 scale

no_show: 0 or 1 (binary classification)

Known Data Issues:

Some age values are negative or >115 (filter out)

Patient_id appears as scientific notation in some cases

Scheduled_day can be AFTER appointment_day (data error - filter)

ML Preprocessing Steps Needed:

Filter age between 0 and 115

Filter lead_days >= 0 (scheduled before appointment)

Handle missing values (minimal in this dataset)

Scale numeric features (StandardScaler)

Model Requirements:

Must output probability (0-1) for risk scoring

Must work with action engine thresholds (0.6, 0.8)

Feature importance must be extractable


Commands to Run:
bash
cd Desktop/MediReach
conda activate stable
jupyter notebook
# Open notebooks/02_feature_engineering.ipynb
Success Criteria
Minimum Viable Product (MVP)
MySQL with 3 tables and 5 views

Logistic regression model (AUC > 0.70)

Action engine generating daily queue

Power BI dashboard with 3 pages

Streamlit app deployed with live link

Stretch Goals (If Time Permits)
Random Forest model for comparison

Foreign key constraints added

Historical trend analysis in dashboard

SMS integration simulation



Document Version: 1.0

