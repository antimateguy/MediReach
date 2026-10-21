"""Synthetic data in the raw Kaggle layout, small enough to reason about by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd

NEIGHBOURHOODS = ("CENTRO", "JARDIM", "PRAIA", "MORRO", "ILHA")
FIRST_DAY = "2016-05-02"
N_DAYS = 20
TIE_PATIENT_INDEX = 0


def patient_id_for(index: int) -> int:
    """Large ids, like the real file, to exercise the float-to-integer step."""
    return 10_000_000_000_000 + index * 1_000


def make_raw_appointments(n_patients: int = 40, n_appointments: int = 200, seed: int = 0) -> pd.DataFrame:
    """About 200 appointments for about 40 patients over 20 business days.

    The first four rows are fixed so that tests have a known case: patient 0
    has one early visit, two visits on the same later day (a tie), and one
    visit after that.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(FIRST_DAY, periods=N_DAYS)
    patient_index = rng.integers(0, n_patients, n_appointments)
    day_index = rng.integers(0, N_DAYS, n_appointments)
    patient_index[:4] = TIE_PATIENT_INDEX
    day_index[:4] = [1, 5, 5, 10]

    lead = np.where(rng.random(n_appointments) < 0.4, 0, rng.integers(1, 41, n_appointments))
    appointment_day = days[day_index]
    seconds = rng.integers(7 * 3600, 20 * 3600, n_appointments)
    scheduled = appointment_day - pd.to_timedelta(lead, unit="D") + pd.to_timedelta(seconds, unit="s")
    no_show = rng.random(n_appointments) < np.where(lead == 0, 0.05, 0.35)

    # Attributes are fixed per patient, as they are (almost always) in the real data.
    attributes = pd.DataFrame({
        "Gender": rng.choice(["F", "M"], n_patients),
        "Age": rng.integers(0, 90, n_patients),
        "Neighbourhood": rng.choice(NEIGHBOURHOODS, n_patients),
        "Scholarship": rng.integers(0, 2, n_patients),
        "Hipertension": rng.integers(0, 2, n_patients),
        "Diabetes": rng.integers(0, 2, n_patients),
        "Alcoholism": rng.integers(0, 2, n_patients),
        "Handcap": rng.choice([0, 0, 0, 1, 2], n_patients),
    }).iloc[patient_index].reset_index(drop=True)

    frame = pd.DataFrame({
        "PatientId": [float(patient_id_for(i)) for i in patient_index],
        "AppointmentID": 5_000_000 + np.arange(n_appointments),
        "ScheduledDay": scheduled.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "AppointmentDay": appointment_day.strftime("%Y-%m-%dT00:00:00Z"),
        "SMS_received": np.where(lead == 0, 0, rng.integers(0, 2, n_appointments)),
        "No-show": np.where(no_show, "Yes", "No"),
    })
    frame = pd.concat([frame, attributes], axis=1)
    return frame[RAW_COLUMNS]


RAW_COLUMNS = [
    "PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay", "Age", "Neighbourhood",
    "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received", "No-show",
]


def with_extra_row(raw: pd.DataFrame, **changes: object) -> pd.DataFrame:
    """Copy of `raw` plus one more row: the first row with `changes` applied."""
    extra = raw.iloc[[0]].copy()
    extra["AppointmentID"] = int(raw["AppointmentID"].max()) + 1
    for column, value in changes.items():
        extra[column] = value
    return pd.concat([raw, extra], ignore_index=True)
