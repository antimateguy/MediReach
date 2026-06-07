# MediReach - Healthcare Operations Intelligence

**Live Demo:** [Streamlit URL after deployment]
**Status:** MVP Complete (Day 3 of 7)

## Quick Start
1. Clone repo
2. `pip install -r requirements.txt`
3. Configure MySQL (see SYSTEM_ARCHITECTURE.md)
4. `python scripts/load_data.py`
5. `python scripts/train_model.py`
6. `python scripts/generate_queue.py`
7. `streamlit run app.py`

## Results
- 0.742 ROC-AUC predicting patient no-shows
- Catches 38% of no-shows with 57% precision
- Reduces manual outreach by 60% vs random

## Tech Stack
MySQL 8.0 | Python 3.10 | scikit-learn | Power BI | Streamlit

## Architecture
[Need to Insert data flow diagram]



Predict patient no-shows with 0.74 ROC-AUC. Reduces manual outreach by 60%.


## Results
- **ROC-AUC:** 0.742 (exceeds 0.70 target)
- **Recall at threshold 0.6:** 0.381
- **Precision:** 0.568

[Full documentation in SYSTEM_ARCHITECTURE.md]
EOF