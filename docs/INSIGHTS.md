# Insights: what drives no-shows, and what to do about it

**Key question:** given capacity for K contacts per day, which appointments should the clinic contact
to prevent the most no-shows?

Every number below comes from `artifacts/hypotheses.json`, written by
`python -m medireach.pipelines.insights`, which reads the SQL views in `sql/views.sql`.
Base: 110,516 appointments, 62,296 patients, overall no-show rate 20.2%.

## How to read this page

- **Rate** = share of appointments where the patient did not come.
- **95% CI** = Wilson confidence interval for a rate. If two intervals do not overlap, the gap is
  very unlikely to be chance.
- **Test** = two-proportion z-test for two groups, chi-square test for several groups.
- **Verdict rule** (fixed before looking at results, in `medireach/data/hypotheses.py`):
  *Supported* = statistically significant (p < 0.05), in the predicted direction, **and** at least
  2 percentage points. *Partly supported* = real but smaller than 2 points, or only an association
  where the claim is causal. *Not supported* = otherwise.
  The 2-point bar exists because with 110,000 rows almost any gap is "significant"; the question for
  a clinic is whether it is big enough to act on.

## Summary

| ID | Hypothesis | Verdict | Headline |
|---|---|---|---|
| H1 | Longer lead time raises risk | Supported | 32.1% for 8+ days ahead vs 24.1% for 1-7 days; same-day 4.6% |
| H2 | A prior no-show predicts another | Supported | 26.9% with a prior no-show vs 18.8% without |
| H3 | SMS reminders reduce no-shows | Partly supported | Raw data says SMS is worse (+10.9 pts); within the same lead bucket it is better (-4.8 pts) |
| H4 | Age groups differ | Supported | 24.0% for ages 18-34, 15.5% for 65+ |
| H5 | Day of week matters | Partly supported | Real but small: 1.9 pts between best and worst weekday |
| H6 | Clinic locations (the `Neighbourhood` column) differ beyond chance | Supported | 14.6% to 28.9%; 10 of 74 differ after a strict correction |

---

## H1. Bookings made 8+ days ahead no-show a third more often; same-day bookings almost never do

**Evidence** (`v_lead_time`):

| Lead time (days) | n | No-show rate | 95% CI |
|---|---|---|---|
| 0 (same day) | 38,561 | 4.6% | 4.4% - 4.9% |
| 1-3 | 14,673 | 22.9% | 22.2% - 23.6% |
| 4-7 | 17,510 | 25.2% | 24.6% - 25.9% |
| 8-14 | 12,025 | 30.5% | 29.7% - 31.3% |
| 15-30 | 17,370 | 32.6% | 31.9% - 33.3% |
| 31+ | 10,377 | 33.0% | 32.1% - 33.9% |

Test: 8+ days (32.1%) vs 1-7 days (24.1%): +7.9 pts, 95% CI +7.3 to +8.6, p < 0.001.
Including same-day bookings in the comparison group the gap is +18.5 pts (32.1% vs 13.5%).
The rate rises with every bucket and flattens after about two weeks.

**Verdict:** Supported.

**So what:** lead time alone splits appointments into a 5% group and a 33% group. One third of all
appointments (same-day) need no outreach at all. Any model has to beat this one-variable rule to be
worth its complexity, which is why the project uses it as the baseline model.

**Action:** spend no call capacity on same-day bookings; start the call list with bookings made
8+ days ahead. Operationally, offer earlier slots where possible, since waiting is itself the risk.

## H2. A patient who has missed before is more likely to miss again

**Evidence** (`v_history_effect`, built on `v_patient_history`; only the 44,230 appointments where
the patient has an earlier appointment in the data):

| Earlier no-shows | n | No-show rate | 95% CI |
|---|---|---|---|
| 0 | 30,119 | 18.8% | 18.4% - 19.3% |
| 1 | 10,472 | 25.0% | 24.2% - 25.9% |
| 2 or more | 3,639 | 32.1% | 30.6% - 33.7% |

Test: any prior no-show (26.9%) vs none (18.8%): +8.0 pts, 95% CI +7.2 to +8.9, p < 0.001.
"Earlier" means strictly earlier dates, so an appointment never counts its own outcome.

**Verdict:** Supported.

**So what:** behaviour repeats, and the effect grows with each missed visit. But it only helps for
40.0% of appointments: the other 60% are from patients with no earlier visit in a six-week window.

**Action:** flag patients with a previous no-show for a call rather than an SMS. With a longer
history (a year instead of six weeks) this signal would cover far more patients; it is the first
thing to improve with more data.

## H3. SMS reminders look harmful in the raw data and helpful once lead time is held fixed

**Evidence, naive** (`v_sms_by_lead`, summed over buckets):

| Group | n | No-show rate |
|---|---|---|
| SMS received | 35,481 | 27.6% |
| No SMS | 75,035 | 16.7% |

Naive gap: **+10.9 pts** (95% CI +10.3 to +11.4, p < 0.001). Read literally, reminders make things worse.

**Why that is wrong:** SMS is not sent at random. It is never sent for same-day bookings (0% of
them), rarely for 1-3 days ahead (6.2%), and to about 60% of bookings made 4+ days ahead. So the SMS
group is mostly long-lead, high-risk bookings, and the no-SMS group contains all 38,561 low-risk
same-day bookings. Lead time drives both who gets an SMS and who no-shows: a confounder.

**Evidence, stratified** (`v_sms_by_lead`, SMS vs no SMS inside each lead bucket):

| Lead time (days) | n with SMS | Rate with SMS | n without | Rate without | Gap (pts) | 95% CI (pts) | p |
|---|---|---|---|---|---|---|---|
| 1-3 | 906 | 21.3% | 13,767 | 23.0% | -1.7 | -4.4 to +1.1 | 0.244 |
| 4-7 | 10,642 | 24.0% | 6,868 | 27.1% | -3.1 | -4.5 to -1.8 | < 0.001 |
| 8-14 | 7,004 | 28.1% | 5,021 | 33.8% | -5.6 | -7.3 to -4.0 | < 0.001 |
| 15-30 | 10,519 | 29.8% | 6,851 | 36.9% | -7.1 | -8.5 to -5.6 | < 0.001 |
| 31+ | 6,410 | 30.2% | 3,967 | 37.5% | -7.3 | -9.2 to -5.4 | < 0.001 |

(The same-day bucket has no SMS rows, so there is nothing to compare.)
SMS patients no-show less in 5 of 5 buckets, significantly in 4. The size-weighted average gap is
**-4.8 pts** (95% CI -5.6 to -4.0). The sign flips: this is Simpson's paradox.

**Verdict:** Partly supported. The direction is as predicted once the confounder is removed, but the
claim is causal and this is observational data. Within a bucket we still do not know why one patient
got an SMS and another did not, so -4.8 pts is an association, not a measured effect.

**So what:** never judge an intervention from a raw comparison when the intervention was targeted.
The stratified result is consistent with reminders helping, and helping more for long-lead bookings.

**Action:** keep SMS as the low-cost tier for medium-risk patients. Measure the true effect with a
randomised pilot (sized on the app's Impact page) before claiming a number.

## H4. Young adults miss most often; patients aged 65+ least

**Evidence** (`v_age_group`):

| Age group | n | No-show rate | 95% CI |
|---|---|---|---|
| 0-17 | 27,340 | 21.9% | 21.4% - 22.4% |
| 18-34 | 24,247 | 24.0% | 23.4% - 24.5% |
| 35-49 | 21,831 | 20.6% | 20.0% - 21.1% |
| 50-64 | 22,685 | 16.7% | 16.2% - 17.2% |
| 65+ | 14,413 | 15.5% | 14.9% - 16.1% |

Tests: ages 18-34 vs everyone else: 24.0% vs 19.1%, +4.8 pts (95% CI +4.2 to +5.4, p < 0.001).
Ages 65+ vs everyone else: 15.5% vs 20.9%, -5.4 pts (95% CI -6.1 to -4.8, p < 0.001).

**Verdict:** Supported.

**So what:** age is a real but moderate signal (about 5 points either way), weaker than lead time
(H1) or history (H2). Children (0-17) are also above average; their attendance depends on a carer.

**Action:** use age as a secondary factor in the risk score, not as a rule on its own. Consider
channel by age: SMS suits younger adults; a call may suit a carer or an older patient better.

## H5. Day of the week makes a real but small difference

**Evidence** (`v_weekday`):

| Weekday | n | No-show rate | 95% CI |
|---|---|---|---|
| Monday | 22,711 | 20.6% | 20.1% - 21.2% |
| Tuesday | 25,638 | 20.1% | 19.6% - 20.6% |
| Wednesday | 25,866 | 19.7% | 19.2% - 20.2% |
| Thursday | 17,244 | 19.3% | 18.8% - 19.9% |
| Friday | 19,018 | 21.2% | 20.7% - 21.8% |
| Saturday (low n) | 39 | 23.1% | 12.6% - 38.3% |

Test: chi-square across Monday-Friday, p < 0.001. Spread between the lowest (Thursday) and the
highest (Friday): 1.9 pts. Saturday is excluded from the test: 39 appointments give an interval
26 points wide, which says nothing.

**Verdict:** Partly supported. Statistically detectable, below the 2-point bar for acting on it.

**So what:** this is the textbook case of "significant is not the same as important". With 110,000
rows a 1.9-point spread has a tiny p-value, but it would not change who gets a call.

**Action:** do not staff outreach by weekday. Leave weekday out of the model apart from a weekend
flag, and say plainly that the data cannot speak about weekends.

## H6. Clinic locations differ by more than chance, but most individual gaps are not reliable

`Neighbourhood` is where the appointment takes place (the health unit's location), not where the
patient lives; the dataset's own data dictionary says so. So this is a comparison of clinic sites.

**Evidence** (`v_clinic`: 74 of 81 clinic locations have at least 100 appointments):

- Rates run from 14.6% (Mario Cypreste, n = 371) to 28.9% (Santos Dumont, n = 1,276): a 14.4-point
  spread. Chi-square across the 74: p < 0.001.
- 30 of 74 have a 95% CI that excludes the overall rate of 20.2%. Testing 74 groups at once, about 4
  would do so by chance alone.
- After a Bonferroni correction (each clinic must clear a 0.05 / 74 bar) **10 remain**:

| Clinic location | n | No-show rate | Gap to overall (pts) |
|---|---|---|---|
| Santos Dumont | 1,276 | 28.9% | +8.7 |
| Santa Cecilia | 448 | 27.5% | +7.3 |
| Santa Clara | 506 | 26.5% | +6.3 |
| Itarare | 3,514 | 26.3% | +6.1 |
| Jesus de Nazareth | 2,853 | 24.4% | +4.2 |
| Ilha do Principe | 2,266 | 23.5% | +3.3 |
| Caratoira | 2,565 | 23.0% | +2.9 |
| Santo Antonio | 2,744 | 17.6% | -2.6 |
| Jardim da Penha | 3,877 | 16.3% | -3.9 |
| Santa Martha | 3,131 | 15.8% | -4.3 |

(Names are written without accents here; the database keeps the original spelling.)

**Verdict:** Supported.

**So what:** location matters, but the list of "bad" clinic locations shrinks from 30 to 10 when the
multiple-comparison problem is taken seriously. Ranking 74 areas and reacting to the top of the list
would mostly chase noise. Because it is a site, a gap points at the clinic (opening hours, transport
links, how bookings are made there) at least as much as at its patients.

**Action:** use the 7 reliably high-rate sites to look for access barriers (transport, clinic hours),
which is a service question rather than a patient-scoring one. Location is kept out of the risk model
in this version (74 extra categories for a modest, unproven gain); v3 tests a leakage-safe encoding of
it on the validation period and keeps it only if it clears the simplicity margin.

---

## What this means for the outreach decision

1. Lead time and patient history are the two strongest signals; both are known the evening before.
2. Roughly a third of appointments (same-day) can be excluded from outreach at almost no risk.
3. A one-variable rule (lead time) already separates 5% from 33%. The model is judged against that
   rule, not against zero.

## Causality caveat

This is observational data. Everything above describes who no-shows, not what would happen if the
clinic intervened. The model predicts **risk**, not **who responds to a call**. Any benefit estimate
therefore rests on an assumed effectiveness of outreach, and the right way to measure the real effect
is a randomised pilot (the app's Impact page sizes one).

## Responsible use

Scholarship (a welfare-programme indicator) and handicap are used only to prioritise *supportive*
outreach, such as a reminder call or help with transport. They must never be used to restrict access,
deprioritise care, or overbook slots against particular patients. Model performance is reported by
gender and age group (`artifacts/segment_metrics.csv`) to check that it does not work much worse for one group.
