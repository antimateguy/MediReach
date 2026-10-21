-- MediReach analytical views (SQLite dialect, needs SQLite 3.28+ for window frames).
-- Rates are proportions between 0 and 1. Confidence intervals are added in Python
-- (medireach/data/analytics.py), because SQLite has no statistical functions.
-- Bucket edges here must match Settings.lead_bucket_edges / age_group_edges; a test checks it.

-- 1. v_appointment_base: one row per appointment with patient and clinic attributes and the
--    derived groupings every other view uses. Defining buckets once avoids drift.
CREATE VIEW v_appointment_base AS
SELECT
    a.appointment_id,
    a.patient_id,
    a.clinic_id,
    c.name AS clinic,
    a.scheduled_date,
    a.appointment_date,
    a.lead_days,
    a.sms_received,
    a.no_show,
    p.gender,
    p.age,
    p.scholarship,
    p.hypertension,
    p.diabetes,
    p.alcoholism,
    p.handicap,
    CASE WHEN a.lead_days = 0 THEN 1 ELSE 0 END AS is_same_day,
    CASE
        WHEN a.lead_days = 0   THEN '0'
        WHEN a.lead_days <= 3  THEN '1-3'
        WHEN a.lead_days <= 7  THEN '4-7'
        WHEN a.lead_days <= 14 THEN '8-14'
        WHEN a.lead_days <= 30 THEN '15-30'
        ELSE '31+'
    END AS lead_bucket,
    CASE
        WHEN a.lead_days = 0   THEN 1
        WHEN a.lead_days <= 3  THEN 2
        WHEN a.lead_days <= 7  THEN 3
        WHEN a.lead_days <= 14 THEN 4
        WHEN a.lead_days <= 30 THEN 5
        ELSE 6
    END AS lead_bucket_order,
    CASE
        WHEN p.age <= 17 THEN '0-17'
        WHEN p.age <= 34 THEN '18-34'
        WHEN p.age <= 49 THEN '35-49'
        WHEN p.age <= 64 THEN '50-64'
        ELSE '65+'
    END AS age_group,
    CASE
        WHEN p.age <= 17 THEN 1
        WHEN p.age <= 34 THEN 2
        WHEN p.age <= 49 THEN 3
        WHEN p.age <= 64 THEN 4
        ELSE 5
    END AS age_group_order,
    -- strftime('%w') gives 0 = Sunday ... 6 = Saturday.
    CAST(strftime('%w', a.appointment_date) AS INTEGER) AS weekday_number,
    CASE strftime('%w', a.appointment_date)
        WHEN '0' THEN 'Sunday'
        WHEN '1' THEN 'Monday'
        WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday'
        WHEN '5' THEN 'Friday'
        ELSE 'Saturday'
    END AS weekday
FROM appointments AS a
JOIN patients AS p ON p.patient_id = a.patient_id
JOIN clinics AS c ON c.clinic_id = a.clinic_id;

-- 2. v_lead_time: no-show rate by how far ahead the appointment was booked (H1).
CREATE VIEW v_lead_time AS
SELECT
    lead_bucket,
    lead_bucket_order AS sort_key,
    COUNT(*)                      AS n,
    SUM(no_show)                  AS no_shows,
    1.0 * SUM(no_show) / COUNT(*) AS no_show_rate
FROM v_appointment_base
GROUP BY lead_bucket, lead_bucket_order
ORDER BY sort_key;

-- 3. v_age_group: no-show rate by age group (H4).
CREATE VIEW v_age_group AS
SELECT
    age_group,
    age_group_order AS sort_key,
    COUNT(*)                      AS n,
    SUM(no_show)                  AS no_shows,
    1.0 * SUM(no_show) / COUNT(*) AS no_show_rate
FROM v_appointment_base
GROUP BY age_group, age_group_order
ORDER BY sort_key;

-- 4. v_weekday: no-show rate by day of the week (H5). Weekend rows are tiny.
CREATE VIEW v_weekday AS
SELECT
    weekday,
    weekday_number AS sort_key,
    COUNT(*)                      AS n,
    SUM(no_show)                  AS no_shows,
    1.0 * SUM(no_show) / COUNT(*) AS no_show_rate
FROM v_appointment_base
GROUP BY weekday, weekday_number
ORDER BY sort_key;

-- 5. v_clinic: clinic locations with at least 100 appointments, ranked by
--    no-show rate, with the gap to the overall rate (H6).
--    The overall rate is a window over ALL clinics, so it is computed
--    before the n >= 100 filter; the rank is computed after it.
CREATE VIEW v_clinic AS
WITH by_clinic AS (
    SELECT
        clinic,
        COUNT(*)     AS n,
        SUM(no_show) AS no_shows
    FROM v_appointment_base
    GROUP BY clinic
),
with_overall AS (
    SELECT
        clinic,
        n,
        no_shows,
        1.0 * no_shows / n                          AS no_show_rate,
        1.0 * SUM(no_shows) OVER () / SUM(n) OVER () AS overall_rate
    FROM by_clinic
)
SELECT
    clinic,
    n,
    no_shows,
    no_show_rate,
    overall_rate,
    no_show_rate - overall_rate              AS diff_to_overall,
    RANK() OVER (ORDER BY no_show_rate DESC) AS rate_rank
FROM with_overall
WHERE n >= 100
ORDER BY rate_rank, clinic;

-- 6. v_sms_by_lead: lead bucket x SMS flag (H3). SMS goes mostly to longer-lead
--    bookings, so SMS vs no-SMS must be compared within a lead bucket.
CREATE VIEW v_sms_by_lead AS
SELECT
    lead_bucket,
    lead_bucket_order AS sort_key,
    sms_received,
    COUNT(*)                      AS n,
    SUM(no_show)                  AS no_shows,
    1.0 * SUM(no_show) / COUNT(*) AS no_show_rate
FROM v_appointment_base
GROUP BY lead_bucket, lead_bucket_order, sms_received
ORDER BY sort_key, sms_received;

-- 7. v_patient_history: what was known about a patient before each appointment.
--    The RANGE frame ends "1 PRECEDING" in days, so it covers strictly earlier
--    dates only: two appointments on the same day never see each other's outcome.
--    previous_no_show (LAG) is the outcome of the row just before in date/id
--    order; it is for analysis only, because on a same-day tie it would not have
--    been known in time.
CREATE VIEW v_patient_history AS
SELECT
    appointment_id,
    patient_id,
    appointment_date,
    no_show,
    ROW_NUMBER() OVER visit_order         AS visit_number,
    COUNT(*) OVER earlier_dates           AS prior_appointments,
    COALESCE(SUM(no_show) OVER earlier_dates, 0) AS prior_no_shows,
    LAG(no_show) OVER visit_order         AS previous_no_show
FROM appointments
WINDOW
    visit_order AS (PARTITION BY patient_id ORDER BY appointment_date, appointment_id),
    earlier_dates AS (
        PARTITION BY patient_id
        ORDER BY julianday(appointment_date)
        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    );

-- 8. v_history_effect: among appointments where the patient has history, the
--    no-show rate by number of earlier no-shows (H2).
CREATE VIEW v_history_effect AS
WITH with_history AS (
    SELECT
        no_show,
        CASE
            WHEN prior_no_shows = 0 THEN '0'
            WHEN prior_no_shows = 1 THEN '1'
            ELSE '2+'
        END AS prior_no_show_bucket,
        MIN(prior_no_shows, 2) AS sort_key
    FROM v_patient_history
    WHERE prior_appointments > 0
)
SELECT
    prior_no_show_bucket,
    sort_key,
    COUNT(*)                      AS n,
    SUM(no_show)                  AS no_shows,
    1.0 * SUM(no_show) / COUNT(*) AS no_show_rate
FROM with_history
GROUP BY prior_no_show_bucket, sort_key
ORDER BY sort_key;

-- 9. v_weekly_trend: weekly volume and no-show rate with week-over-week change.
--    week_start is the Monday of the appointment's week.
CREATE VIEW v_weekly_trend AS
WITH weekly AS (
    SELECT
        date(appointment_date, 'weekday 0', '-6 days') AS week_start,
        COUNT(*)                      AS n,
        SUM(no_show)                  AS no_shows,
        1.0 * SUM(no_show) / COUNT(*) AS no_show_rate
    FROM appointments
    GROUP BY week_start
)
SELECT
    week_start,
    n,
    no_shows,
    no_show_rate,
    LAG(n) OVER (ORDER BY week_start)                           AS previous_n,
    n - LAG(n) OVER (ORDER BY week_start)                       AS n_change,
    no_show_rate - LAG(no_show_rate) OVER (ORDER BY week_start) AS rate_change
FROM weekly
ORDER BY week_start;
