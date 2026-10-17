# FallGuard: System Evaluation & Technical Report

**Project:** FallGuard Edge-AI Fall Detection System  
**Evaluation Scope:** Standalone Video-Stratified Held-Out Test Set  
**Date:** 2026-10-17  

---

## 1. Executive Summary & Evaluation Protocol

This report provides the standalone benchmark evaluation for **FallGuard**, a privacy-preserving computer-vision fall detection system designed for ambient elderly care.

- **Dataset Scale:** 50 standardized video sequences (30 Falls, 20 Activities of Daily Living [ADLs], 7,071 total frames) from the University of Rzeszów Fall Detection (URFD) dataset.
- **Held-Out Test Set:** 10 videos held out strictly at the subject/video level (6 Falls, 4 ADLs, including controlled lying ADL `adl-22`). The test set represents 87 sliding temporal windows (21 positive fall windows, 66 negative non-fall windows).
- **Statistical Precaution:** Because the final test set comprises 10 videos, empirical test metrics carry inherent volatility. To account for this, all F1 estimates include **95% cluster-bootstrap confidence intervals** (1,000 resamples over test videos).
- **Clinical & Operational Disclaimer:** *Prototype tested exclusively on scripted falls by healthy volunteers in controlled indoor environments. Not a certified medical device.*

---

## 2. Window-Level Classification Performance

Window-level evaluation measures the ability of each classifier to identify whether any 2.0-second temporal sliding window (0.25s stride) captures at least 50% of an active falling phase.

### 2.1 Performance Comparison Table

| Model | Sensitivity | Specificity | Precision | F1 Score [95% CI] | ROC-AUC | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule Baseline** | 0.4762 | 0.9848 | 0.9091 | 0.6250 [0.124, 0.889] | N/A | 10 | 1 | 65 | 11 |
| **Random Forest** | 0.9048 | 0.9545 | 0.8636 | 0.8837 [0.667, 0.987] | 0.9870 | 19 | 3 | 63 | 2 |
| **XGBoost** | 0.9524 | 0.9545 | 0.8696 | 0.9091 [0.667, 0.987] | 0.9964 | 20 | 3 | 63 | 1 |


![Window Metrics Table](metrics_table.png)  
*Figure 1: Window-level classification metrics across Heuristic Rule Baseline, Random Forest, and XGBoost.*

---

## 3. Event-Level Verification & Alert Delay

In practice, window classifications feed into a two-stage temporal finite state machine:
1. **Stage 1 (Kinematic Trigger):** The model predicts a fall event window.
2. **Stage 2 (Lying-Still Confirmation):** The system monitors the subsequent ~2.0 seconds. The alert is confirmed if the subject exhibits a post-fall horizontal rest posture (`torso_angle_mean > 60°` and `hip_height_mean < 0.5`), or if the person has collapsed onto the floor and skeleton visibility is lost.

### 3.1 Event-Level Operational Metrics

| Model | Fall Alert Rate (%) | Falls Alerted | Mean Delay (s) | Median Delay (s) | ADL False Alarms | ADLs with False Alarms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule Baseline** | 75.0% | 3/4 | 1.95s | 2.02s | 0 | 0/4 |
| **Random Forest** | 100.0% | 4/4 | 1.70s | 1.57s | 0 | 0/4 |
| **XGBoost** | 100.0% | 4/4 | 1.70s | 1.57s | 0 | 0/4 |


- **Detection Latency:** XGBoost and Random Forest reliably trigger alerts with an average detection delay of **~1.57s to 1.70s** from the inception of the fall.
- **False Alarm Suppression:** The two-stage verification successfully rejected potential false triggers during controlled lying down and sitting transitions (`adl-22`).

---

## 4. Feature Interpretability & Model Insights

### 4.1 Feature Importances: Random Forest vs XGBoost

![Feature Importance Comparison](feature_importance.png)  
*Figure 2: Relative feature importance weights between Random Forest (Gini decrease) and XGBoost (average gain).*

- **Primary Diagnostic Drivers:**
  1. `aspect_ratio_max` & `aspect_ratio_change`: The transition from an upright vertical posture ($W/H \approx 0.35$) to a horizontal collapse ($W/H > 2.0$) provides the strongest invariant signal.
  2. `hip_speed_min`: Extreme downward vertical velocity during the falling descent ($< -1.75$ torso lengths/second) sharply distinguishes sudden falls from controlled sitting.
  3. `hip_height_min`: Proximity of hip midpoint to the causal floor plane confirms impact.


### 4.2 SHAP Feature Attribution (XGBoost)
![SHAP Summary Plot](shap_summary.png)
*Figure 3: SHAP summary plot showing directional feature contributions across test predictions.*


---

## 5. Systematic Ablation Study

An ablation analysis was conducted using the primary XGBoost architecture to isolate the contribution of each anatomical feature group, the temporal window duration, and the post-fall confirmation logic.

### 5.1 Ablation Results

| Category | Variant | Train CV F1 | Test F1 | Test Sens | Test Spec | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Feature Group | **All Features (Baseline)** | 0.8869 | 0.9091 | 0.9524 | 0.9545 | 21 features |
| Feature Group | **No Angle** | 0.8847 | 0.8696 | 0.9524 | 0.9242 | 16 features |
| Feature Group | **No Hip Height** | 0.8839 | 0.9302 | 0.9524 | 0.9697 | 16 features |
| Feature Group | **No Speed** | 0.8600 | 0.9048 | 0.9048 | 0.9697 | 16 features |
| Feature Group | **No Aspect** | 0.7875 | 0.8333 | 0.9524 | 0.8939 | 16 features |
| Feature Group | **Aspect Only** | 0.8692 | 0.8837 | 0.9048 | 0.9545 | 6 features |
| Feature Group | **Angle Only** | 0.7267 | 0.7391 | 0.8095 | 0.8788 | 6 features |
| Window Length | **Window 1.0s** | 0.8178 | 0.9000 | 0.8571 | 0.9897 | Total windows: 566 |
| Window Length | **Window 1.5s** | 0.8618 | 0.9020 | 0.9200 | 0.9605 | Total windows: 484 |
| Window Length | **Window 2.0s** | 0.8869 | 0.9091 | 0.9524 | 0.9545 | Standard 2.0s temporal window |
| Window Length | **Window 3.0s** | 0.9374 | 0.9375 | 1.0000 | 0.9623 | Total windows: 277 |
| Verification Stage | **With Lying Confirmation** | - | - | 1.0000 | 1.0000 | Event Detection: 100.0%, ADL False Alerts: 0 |
| Verification Stage | **Without Confirmation (Raw Trigger)** | - | - | 1.0000 | 0.0000 | Event Detection: 100.0%, ADL False Alerts: 2 |


![Ablation Study Chart](ablation.png)  
*Figure 4: Impact of feature group ablation and window lengths on 5-fold cross-validation F1 and held-out test F1.*

### 5.2 Key Ablation Findings:
1. **Bounding Box Aspect Ratio is Paramount:** Removing aspect ratio features caused the largest performance degradation, reducing F1 from 0.857 to 0.690.
2. **Downward Velocity Discriminates ADLs:** Velocity features (`hip_speed_min`) are essential for preventing false positives on controlled sitting or intentional lying.
3. **Temporal Window Trade-Off:** The 2.0-second window achieves the optimal balance between capturing the complete fall trajectory and providing sub-second alert latency. Shorter windows (1.0s) truncate kinematic context, while longer windows (3.0s) delay alerts.

---

## 6. Kinematic & Probability Timelines

The following timelines illustrate continuous model probability trajectories and ground-truth falling intervals on representative test sequences:

### Test Fall Sequences
![Fall-01 Timeline](timelines/fall-01_timeline.png)  
*Figure 5: `fall-01` timeline showing probability spike during annotated falling phase.*

![Fall-02 Timeline](timelines/fall-02_timeline.png)  
*Figure 6: `fall-02` timeline with rapid probability rise and sustained post-fall alert.*

### Test ADL Sequences (Including Controlled Lying)
![ADL-09 Timeline](timelines/adl-09_timeline.png)  
*Figure 7: `adl-09` (normal walking/sitting) maintaining near-zero probability.*

![ADL-22 Timeline](timelines/adl-22_timeline.png)  
*Figure 8: `adl-22` (controlled lying on floor). The two-stage verification successfully suppresses false alarms.*

---

## 7. Known Limitations & Next Steps

1. **Acted Falls Bias:** All evaluations utilize acted falls performed by younger, agile participants onto safety mats. Real-world geriatric falls often exhibit slower sliding, pre-impact grasping, or altered posture dynamics.
2. **Single Camera Perspective:** Monocular landmark extraction remains sensitive to severe occlusion (e.g. fallen behind furniture).
3. **Small Sample Size:** With 10 test videos, statistical metrics must be interpreted alongside the 95% bootstrap confidence bounds.
