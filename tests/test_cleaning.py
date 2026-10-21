"""Phase 2: cleaning rules and the CleaningReport."""
from __future__ import annotations

import pandas as pd
import pytest

from medireach.config import Settings
from medireach.data.cleaning import RawDataLoader, clinic_table
from medireach.errors import DataAccessError
from tests.factories import TIE_PATIENT_INDEX, patient_id_for, with_extra_row


def _clean(settings: Settings, raw: pd.DataFrame):
    return RawDataLoader(settings).clean(raw)


def test_same_day_bookings_are_kept(settings: Settings, raw: pd.DataFrame) -> None:
    same_day_in = int((raw["ScheduledDay"].str[:10] == raw["AppointmentDay"].str[:10]).sum())
    _, appointments, report = _clean(settings, raw)
    assert same_day_in > 0
    assert int((appointments["lead_days"] == 0).sum()) == same_day_in
    assert report.same_day_rows == same_day_in


def test_booking_made_in_the_evening_for_today_has_lead_zero(settings: Settings, raw: pd.DataFrame) -> None:
    # The v1 bug: 18:38 on the day compared against midnight looked "after the appointment".
    probe = with_extra_row(raw, ScheduledDay="2016-05-10T18:38:08Z", AppointmentDay="2016-05-10T00:00:00Z")
    _, appointments, report = _clean(settings, probe)
    row = appointments[appointments["appointment_id"] == probe["AppointmentID"].max()]
    assert len(row) == 1 and int(row["lead_days"].iloc[0]) == 0
    assert report.dropped["negative_lead_days"] == 0


def test_negative_lead_rows_are_dropped(settings: Settings, raw: pd.DataFrame) -> None:
    probe = with_extra_row(raw, ScheduledDay="2016-05-11T08:00:00Z", AppointmentDay="2016-05-10T00:00:00Z")
    _, appointments, report = _clean(settings, probe)
    assert report.dropped["negative_lead_days"] == 1
    assert (appointments["lead_days"] >= 0).all()
    assert len(appointments) == len(raw)


def test_invalid_ages_are_dropped(settings: Settings, raw: pd.DataFrame) -> None:
    probe = with_extra_row(with_extra_row(raw, Age=-1), Age=115)
    patients, _, report = _clean(settings, probe)
    assert report.dropped["invalid_age"] == 2
    assert patients["age"].between(0, settings.max_age).all()


def test_duplicate_appointment_ids_are_dropped(settings: Settings, raw: pd.DataFrame) -> None:
    probe = pd.concat([raw, raw.iloc[[3]]], ignore_index=True)
    _, appointments, report = _clean(settings, probe)
    assert report.dropped["duplicate_appointment_id"] == 1
    assert appointments["appointment_id"].is_unique


def test_dropped_counts_account_for_every_removed_row(settings: Settings, raw: pd.DataFrame) -> None:
    probe = with_extra_row(raw, Age=-1)
    probe = with_extra_row(probe, ScheduledDay="2016-06-30T08:00:00Z")
    probe = pd.concat([probe, probe.iloc[[7]]], ignore_index=True)
    _, appointments, report = _clean(settings, probe)
    assert report.rows_in == len(probe)
    assert report.rows_out == len(appointments)
    assert sum(report.dropped.values()) == report.rows_in - report.rows_out == 3


def test_patient_ids_become_integers_without_collisions(settings: Settings, raw: pd.DataFrame) -> None:
    patients, appointments, report = _clean(settings, raw)
    assert report.patient_id_collisions == 0
    assert str(patients["patient_id"].dtype) == "int64"
    assert patients["patient_id"].nunique() == raw["PatientId"].nunique() == report.patients
    assert set(appointments["patient_id"]) == set(patients["patient_id"])


def test_target_is_encoded_as_zero_and_one(settings: Settings, raw: pd.DataFrame) -> None:
    _, appointments, report = _clean(settings, raw)
    assert set(appointments["no_show"].unique()) <= {0, 1}
    assert appointments["no_show"].sum() == (raw["No-show"] == "Yes").sum()
    assert report.base_rate_before == pytest.approx((raw["No-show"] == "Yes").mean())


def test_patient_attributes_come_from_the_most_recent_appointment(settings: Settings, raw: pd.DataFrame) -> None:
    # Patient 0's last visit in the fixture is row 3; give it a different age.
    probe = raw.copy()
    probe.loc[:2, "Age"] = 40
    probe.loc[3, "Age"] = 41
    others = probe.index[(probe["PatientId"] == probe.loc[0, "PatientId"]) & (probe.index > 3)]
    probe = probe.drop(index=others).reset_index(drop=True)
    patients, _, report = _clean(settings, probe)
    patient = patients[patients["patient_id"] == patient_id_for(TIE_PATIENT_INDEX)]
    assert int(patient["age"].iloc[0]) == 41
    assert report.patients_with_varying_attributes == 1
    assert report.varying_by_attribute["age"] == 1
    assert report.varying_by_attribute["gender"] == 0


def test_bad_input_raises_a_named_error(settings: Settings, raw: pd.DataFrame) -> None:
    with pytest.raises(DataAccessError, match="missing columns"):
        _clean(settings, raw.drop(columns=["Handcap"]))
    with pytest.raises(DataAccessError, match="No-show"):
        _clean(settings, with_extra_row(raw, **{"No-show": "Maybe"}))
    with pytest.raises(DataAccessError, match="read_raw_csv"):
        RawDataLoader(settings).load("no/such/file.csv")


def test_cleaning_does_not_modify_its_input(settings: Settings, raw: pd.DataFrame) -> None:
    before = raw.copy(deep=True)
    _clean(settings, raw)
    pd.testing.assert_frame_equal(raw, before)


def test_clinics_come_from_neighbourhood_with_stable_ids(settings: Settings, raw: pd.DataFrame) -> None:
    # The source calls it Neighbourhood; it is where the appointment takes place.
    _, appointments, report = _clean(settings, raw)
    clinics = clinic_table(appointments)
    assert list(clinics["name"]) == sorted(raw["Neighbourhood"].unique())
    assert list(clinics["clinic_id"]) == list(range(1, len(clinics) + 1))
    assert report.clinics == len(clinics)
    shuffled = _clean(settings, raw.sample(frac=1.0, random_state=4))[1]
    assert clinic_table(shuffled).equals(clinics)
