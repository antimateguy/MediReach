# MediReach: Key Business Insights & Recommendations

## Dataset Overview
- **Time Period:** April-June 2016
- **Patients:** 62,299 unique individuals
- **Appointments:** 110,527 scheduled visits
- **Overall No-Show Rate:** 19.64%

---

## Insight 1: Lead Time is the Strongest Predictor (CRITICAL)

| Wait Time | Appointments | No-Show Rate | Risk Multiplier |
|-----------|--------------|--------------|-----------------|
| 0-7 days | 70,748 | 13.52% | 1.0x (Baseline) |
| 8-14 days | 12,025 | 30.47% | **2.3x** |
| 15-21 days | 8,874 | 32.24% | **2.4x** |
| 22-30 days | 8,497 | 32.95% | **2.4x** |
| 30+ days | 10,378 | 33.00% | **2.4x** |

**Business Implication:**
Patients waiting >7 days are 2.3x more likely to no-show. The majority of appointments (70,748) are scheduled within 7 days and have low risk.

**Recommendation:**
- Flag appointments scheduled >7 days out for automatic SMS reminder
- Implement waitlist management to reduce lead time to <7 days
- For 30+ day waits, call patient to confirm or reschedule

---

## Insight 2: Age Segmentation Identifies Target Demographics

| Age Group | Patient Count | Appointments | No-Show Rate | Action Priority |
|-----------|---------------|--------------|--------------|-----------------|
| 18-34 (Young Adults) | 13,434 | 24,237 | **23.97%** | HIGH |
| 0-17 (Children) | 16,262 | 27,398 | 21.91% | MEDIUM |
| 35-49 (Adults) | 11,722 | 21,903 | 20.50% | MEDIUM |
| 50-64 (Seniors) | 12,525 | 22,596 | 16.76% | LOW |
| 65+ (Elderly) | 8,356 | 14,393 | 15.49% | LOWEST |

**Business Implication:**
Young adults (18-34) are highest risk at 24% no-show, while elderly patients (65+) are most reliable at 15.5%.

**Recommendation:**
- Targeted SMS campaigns for 18-34 age group with flexible rescheduling options
- Reduced outreach for elderly (already reliable, save resources)
- Children's appointments: remind parents 24 hours before

---

## Insight 3: Geographic Clustering Requires Local Intervention

**Top 5 High-Risk Neighborhoods:**

| Rank | Neighbourhood | No-Show Rate | vs. Average |
|------|---------------|--------------|--------------|
| 1 | SANTOS DUMONT | 28.92% | +47% higher |
| 2 | SANTA CECÍLIA | 27.46% | +40% higher |
| 3 | SANTA CLARA | 26.48% | +35% higher |
| 4 | ITARARÉ | 26.27% | +34% higher |
| 5 | JESUS DE NAZARETH | 24.40% | +24% higher |

**Business Implication:**
Five neighborhoods show no-show rates 24-47% above the 19.6% city average, suggesting geographic barriers (transportation, awareness, access).

**Recommendation:**
- Deploy community health workers to top 3 neighborhoods
- Investigate transportation barriers in SANTOS DUMONT (highest risk)
- Local language SMS campaigns for these ZIP codes

---

## Insight 4: Patient History Predicts Future Behavior

From `vw_patient_risk_profile`:
- Patients with 1 appointment only: 100% no-show OR 0% no-show (binary)
- Patients with 3+ appointments and >50% no-show rate: **High risk for recurrence**
- Patients with 0% no-show history: Highly reliable (<10% future risk)

**Recommendation:**
- Escalate outreach for patients with 2+ past no-shows
- No outreach needed for patients with perfect attendance history

---

## Action Engine Rules (Final)

| Risk Factor | Priority | Action |
|-------------|----------|--------|
| Lead time >14 days | HIGH | Phone call |
| Age 18-34 + lead time >7 days | HIGH | SMS + email |
| Top 5 neighbourhood | MEDIUM | SMS + community worker |
| 2+ past no-shows | HIGH | Call before each appointment |
| Lead time 0-7 days + age 65+ | LOW | No action |

---

## Estimated Business Impact

**Target Segment:** 30% of appointments (high-risk)
**Expected Reduction:** 25-30% in no-show rate
**Financial Impact (500-bed hospital):**
- Baseline no-shows: 20% of 10,000 monthly appointments = 2,000 misses
- After intervention: 1,400 misses (30% reduction)
- Monthly savings: 600 appointments × ₹1,500 avg visit value = **₹9L/month**
- **Annual savings: ~₹1.08 Cr**

---

## Interview Talking Points

> "Three insights drive this solution:
> 1. **Lead time is 2.3x risk factor** - patients waiting >7 days need intervention
> 2. **Young adults are highest risk** - 24% no-show vs 15.5% for elderly  
> 3. **Five neighborhoods need local outreach** - 27-29% no-show rates

> Our action engine prioritizes calls for >14 day waits, SMS for 7-14 days, and deploys community health workers in top neighborhoods. Estimated annual savings: ₹1 Cr for a 500-bed hospital."