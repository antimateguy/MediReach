

USE medireach;

-- =====================================================
-- VIEW 1: Patient Risk Profile (Window Function)
-- Shows each patient's no-show rate compared to average
-- =====================================================

DROP VIEW IF EXISTS vw_patient_risk_profile;

-- works with strict mode
CREATE OR REPLACE VIEW vw_patient_risk_profile AS
WITH patient_history AS (
    SELECT 
        p.patient_id,
        ANY_VALUE(p.age) AS age,
        ANY_VALUE(p.gender) AS gender,
        ANY_VALUE(p.neighbourhood) AS neighbourhood,
        ANY_VALUE(p.scholarship) AS scholarship,
        ANY_VALUE(p.hypertension) AS hypertension,
        COUNT(a.appointment_id) AS total_appointments,
        SUM(a.no_show) AS no_show_count,
        ROUND(100.0 * SUM(a.no_show) / COUNT(a.appointment_id), 2) AS no_show_rate
    FROM patients p
    JOIN appointments a ON p.patient_id = a.patient_id
    GROUP BY p.patient_id
)
SELECT 
    *,
    ROUND(AVG(no_show_rate) OVER (), 2) AS overall_avg_rate,
    ROUND(no_show_rate - AVG(no_show_rate) OVER (), 2) AS rate_diff_from_avg,
    ROW_NUMBER() OVER (ORDER BY no_show_rate DESC) AS risk_rank
FROM patient_history;

-- =====================================================
-- VIEW 2: Age Group Analysis (Business Insight)
-- =====================================================
CREATE OR REPLACE VIEW vw_age_analysis AS
SELECT 
    CASE 
        WHEN age < 18 THEN '0-17 (Children)'
        WHEN age < 35 THEN '18-34 (Young Adults)'
        WHEN age < 50 THEN '35-49 (Adults)'
        WHEN age < 65 THEN '50-64 (Seniors)'
        ELSE '65+ (Elderly)'
    END AS age_group,
    COUNT(DISTINCT p.patient_id) AS patient_count,
    COUNT(a.appointment_id) AS total_appointments,
    SUM(a.no_show) AS no_shows,
    ROUND(100.0 * SUM(a.no_show) / COUNT(a.appointment_id), 2) AS no_show_rate,
    ROUND(AVG(DATEDIFF(a.appointment_day, a.scheduled_day)), 1) AS avg_lead_days
FROM patients p
JOIN appointments a ON p.patient_id = a.patient_id
GROUP BY age_group
ORDER BY no_show_rate DESC;

-- =====================================================
-- VIEW 3: Lead Time Impact (CRITICAL Insight)
-- Shows: Longer wait = higher no-show
-- =====================================================
CREATE OR REPLACE VIEW vw_lead_time_analysis AS
SELECT 
    CASE 
        WHEN DATEDIFF(appointment_day, scheduled_day) <= 7 THEN 'Week 1 (0-7 days)'
        WHEN DATEDIFF(appointment_day, scheduled_day) <= 14 THEN 'Week 2 (8-14 days)'
        WHEN DATEDIFF(appointment_day, scheduled_day) <= 21 THEN 'Week 3 (15-21 days)'
        WHEN DATEDIFF(appointment_day, scheduled_day) <= 30 THEN 'Month 2 (22-30 days)'
        ELSE '30+ days'
    END AS lead_time_bucket,
    COUNT(*) AS appointments,
    ROUND(100.0 * SUM(no_show) / COUNT(*), 2) AS no_show_rate,
    ROUND(AVG(age), 1) AS avg_patient_age
FROM appointments a
JOIN patients p ON a.patient_id = p.patient_id
WHERE DATEDIFF(appointment_day, scheduled_day) >= 0
GROUP BY lead_time_bucket
ORDER BY 
    CASE lead_time_bucket
        WHEN 'Week 1 (0-7 days)' THEN 1
        WHEN 'Week 2 (8-14 days)' THEN 2
        WHEN 'Week 3 (15-21 days)' THEN 3
        WHEN 'Month 2 (22-30 days)' THEN 4
        ELSE 5
    END;

-- =====================================================
-- VIEW 4: Neighbourhood Risk Ranking (Window Function)
-- =====================================================
CREATE OR REPLACE VIEW vw_neighbourhood_risk AS
SELECT 
    p.neighbourhood,
    COUNT(DISTINCT p.patient_id) AS patients,
    COUNT(a.appointment_id) AS appointments,
    SUM(a.no_show) AS no_shows,
    ROUND(100.0 * SUM(a.no_show) / COUNT(a.appointment_id), 2) AS no_show_rate,
    RANK() OVER (ORDER BY ROUND(100.0 * SUM(a.no_show) / COUNT(a.appointment_id), 2) DESC) AS risk_rank,
    ROUND(AVG(DATEDIFF(a.appointment_day, a.scheduled_day)), 1) AS avg_lead_days
FROM patients p
JOIN appointments a ON p.patient_id = a.patient_id
GROUP BY p.neighbourhood
HAVING COUNT(a.appointment_id) >= 30
ORDER BY risk_rank
LIMIT 20;

-- =====================================================
-- VIEW 5: Monthly Trend with LAG (Compare MoM)
-- =====================================================
CREATE OR REPLACE VIEW vw_monthly_trend AS
SELECT 
    DATE_FORMAT(a.appointment_day, '%Y-%m') AS month,
    COUNT(*) AS appointments,
    ROUND(100.0 * SUM(a.no_show) / COUNT(*), 2) AS no_show_rate,
    LAG(ROUND(100.0 * SUM(a.no_show) / COUNT(*)), 1) OVER (ORDER BY DATE_FORMAT(a.appointment_day, '%Y-%m')) AS prev_month_rate,
    ROUND(100.0 * SUM(a.no_show) / COUNT(*) - 
          LAG(ROUND(100.0 * SUM(a.no_show) / COUNT(*)), 1) OVER (ORDER BY DATE_FORMAT(a.appointment_day, '%Y-%m')), 2) AS change_from_prev_month
FROM appointments a
GROUP BY DATE_FORMAT(a.appointment_day, '%Y-%m')
ORDER BY month;

-- =====================================================
-- VERIFICATION QUERIES (Run these to test)
-- =====================================================

-- Check all views created
-- SHOW FULL TABLES WHERE Table_type = 'VIEW';

-- Test each view
-- SELECT * FROM vw_age_analysis;
-- SELECT * FROM vw_lead_time_analysis;
-- SELECT * FROM vw_neighbourhood_risk;
-- SELECT * FROM vw_monthly_trend;
-- SELECT * FROM vw_patient_risk_profile LIMIT 10;