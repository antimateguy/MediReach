# Data notes

Every number on this page comes from `artifacts/cleaning_report.json` (written by
`python -m medireach.pipelines.build_db`) unless a different command is shown next to it.

## Source

Kaggle "Medical Appointment No Shows" (`data/raw/KaggleV2-May-2016.csv`): 110,527 appointment rows,
14 columns, no missing values. Appointments run from 2016-04-29 to 2016-06-08 (27 distinct appointment
dates) in one Brazilian city. One row is one appointment; a patient can appear many times.

## Cleaning rules (applied in this order)

| # | Rule | Rows removed |
|---|---|---|
| 1 | Rename columns to snake_case and fix the source typos (`Hipertension` -> `hypertension`, `Handcap` -> `handicap`, `No-show` -> `no_show`) | 0 |
| 2 | `patient_id`: float -> integer by rounding | 0 (0 collisions) |
| 3 | `no_show`: "Yes" -> 1, "No" -> 0. Note the direction: 1 means the patient did NOT come | 0 |
| 4 | Parse timestamps; reduce both to dates; `lead_days = appointment_date - scheduled_date` | 0 |
| 5 | Drop `lead_days < 0` (booked after the appointment date: impossible) | 5 |
| 6 | Drop age below 0 or above 110 | 6 |
| 7 | Drop duplicate `appointment_id` | 0 |
| 8 | `handicap` keeps its original 0-4 scale; the yes/no flag `has_handicap` is derived later | 0 |
| 9 | Patient attributes are taken from each patient's most recent appointment | 0 |

Result: **110,527 rows in, 110,516 rows out** (11 removed), **62,296 patients**.
The no-show rate is 20.19% before cleaning and 20.19% after (0.2019 and 0.2019 to four places), so
cleaning did not move the target.

The 6 invalid-age rows are 1 row with age -1 and 5 rows with age 115
(`python` check on the raw file: `(Age < 0).sum()` = 1, `(Age > 110).sum()` = 5).
Age 0 is valid (infants) and is kept.

## The v1 timestamp bug (the most important data fix)

`ScheduledDay` is a full timestamp (for example `2016-04-29T18:38:08Z`). `AppointmentDay` is a date
only: its time is always `00:00:00`. Version 1 filtered with `scheduled_day <= appointment_day`,
comparing a timestamp with midnight. A patient who booked at 18:38 for the same day looked as if they
had booked *after* their appointment, and the row was thrown away.

- That filter removes **38,568 rows (35% of the data)** (recorded in the cleaning report as
  `v1_timestamp_filter_drops`). 38,563 of them are valid same-day bookings;
  only 5 are truly invalid. (Command: on the raw file, count rows where the parsed `ScheduledDay` is
  later than the parsed `AppointmentDay`.)
- It is not a random 35%. Same-day bookings no-show at **4.6%**; everyone else at 28.5%. Dropping
  them turned a 20.2% problem into a 28.5% problem and removed the single clearest pattern in the data.
- Fix: compare dates, not timestamps. Same-day bookings (`lead_days = 0`) are kept: 38,561 rows after
  cleaning, 34.9% of the table. A test pins this
  (`test_booking_made_in_the_evening_for_today_has_lead_zero`).

How to explain it: "The appointment column has no time of day, so comparing it to a timestamp treats
every same-day booking as invalid. I compare calendar dates instead, and I log every dropped row by
rule so a silent 35% loss cannot happen again."

## Patient attributes that vary

The `patients` table needs one value per patient, but the source repeats attributes on every
appointment row. Measured on the cleaned data:

- **1,168 patients (1.9%)** have an attribute that changes between their rows.
- In every case the attribute is **age**, and the change is exactly one year (a birthday inside the
  six-week window). Gender, scholarship and the health flags never vary. (The clinic is an attribute
  of the appointment, not the patient, and is stored on `appointments`.)

Decision: take attributes from the most recent appointment. For those 1,168 patients an earlier
appointment therefore shows an age one year too high. That is negligible for five-way age groups and
is simpler than storing age per appointment.

## Schema

`sql/schema.sql` creates `clinics` (primary key `clinic_id`), `patients` (primary key `patient_id`),
`appointments` (primary key `appointment_id`, foreign keys to `patients` and `clinics`), and the two
outreach tables. Clinic ids number the 81 sites in alphabetical order, so they are stable between
builds. CHECK constraints enforce the cleaning rules inside the database (`lead_days >= 0`, age 0-110,
flags 0/1). Foreign keys are switched on for every connection. Dates are stored as ISO text
(`YYYY-MM-DD`), which sorts correctly.

The build is reproducible: running `build_db` twice produces a byte-identical `data/medireach.db`.

## Known quirks (each one affects how results should be read)

| Quirk | Evidence | Consequence |
|---|---|---|
| SMS is confounded with lead time | SMS is never sent for same-day bookings (0%), rarely for 1-3 days ahead (6.2%), and to about 60% of bookings made 4+ days ahead | A raw SMS vs no-SMS comparison is misleading; see H3 in `INSIGHTS.md` |
| Weekend rows are tiny | 39 Saturday appointments, 0 Sunday | No conclusion about weekends is possible |
| Six-week window | 27 appointment dates | Patient history is thin: only 40.0% of appointments (44,230) have any earlier appointment in the data. History that happened before 2016-04-29 is invisible |
| Volume is uneven by week | The week of 2016-05-23 has 7,918 appointments against about 21,000 in a normal week | Weekly trend comparisons are not like-for-like |
| Timestamps are labelled UTC (`Z`) | Source file | The date is used exactly as written; no time-zone shift is applied |
| One city, 2016 | Source | Results describe this clinic network, not outpatient care in general |
| `no_show = 1` means absent | Source column "No-show" | Easy to read backwards; every rate in this project is a no-show rate |
| `Neighbourhood` is the clinic's location, not the patient's | The dataset's data dictionary: "where the appointment takes place"; its version notes: "the place where the hospital is located and not where the patient lives" | It describes the site, so a gap points at the clinic as much as at its patients. Stored as a `clinics` table (81 rows) with `clinic_id` on each appointment. (v2 first stored it on `patients`, which happened to be harmless: no patient ever changed site.) |
| Neighbourhood names contain accents | Source | Stored as UTF-8; 81 sites, 7 of them with fewer than 100 appointments |
| `SMS_received` = one or more messages | Data dictionary: "1 or more messages sent to the patient" | The flag says a reminder was sent, not how many |
