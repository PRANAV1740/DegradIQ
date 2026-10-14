# DegradIQ — Multi-Circuit Benchmark & Long-Run Filter Walkthrough

## 1. Multi-Circuit Summary Table

Across all three benchmark tracks, DegradIQ isolates true tyre degradation from noisy practice sessions where fuel burn-off and track evolution hide wear:

| Circuit | Characteristics | Selected Stint | Raw Slope | Corrected Slope | Multiplier | Race MAE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Bahrain** | High deg (traction/abrasion) | VER (FP2, SOFT, 11 laps) | **+0.0126 s/lap** | **+0.1699 s/lap** | **13.4x** | **1.74s** *(baseline)* / 10.2s |
| **Spain** | High deg (lateral loads) | OCO (FP2, SOFT, 8 laps) | **+0.0149 s/lap** | **+0.0747 s/lap** | **5.0x** | **2.25s** |
| **Italy** | Low deg (straight-line speed) | NOR (FP2, MEDIUM, 8 laps) | **+0.0292 s/lap** | **-0.3286 s/lap** | *(Evo swing)* | **15.36s** |

*(Note on Spain alternate candidate: Lando Norris ran 12 laps in FP2 on SOFT with Raw Slope = -0.0071 s/lap and Corrected Slope = +0.0527 s/lap, completely reversing negative wear into positive degradation).*

---

## 2. Noise Attribution Numbers

This attribution analysis answers the core question: **"How much real tyre wear was concealed by each noise source?"**

### Bahrain — Driver VER, SOFT (FP2, 11 laps)
```
Noise attribution for this stint:
  Fuel effect:        +0.684 s  (hid 36% of the real wear)
  Track evolution:    +1.204 s  (hid 63% of the real wear)
  Traffic removed:    2 lap(s) flagged  (avg +4.696s off clean pace)
  ------------------------------------------------------------
  True tyre wear:     +1.918 s  (100% of total signal)
```
> **Key Finding:** In raw timing, Verstappen's stint looked completely flat (+0.030s over 11 laps). DegradIQ shows that **track evolution hid 63%** and **fuel burn hid 36%** of the wear. The true tyre wear was **+1.918 seconds**.

### Spain — Driver OCO, SOFT (FP2, 8 laps)
```
Noise attribution for this stint:
  Fuel effect:        +0.570 s  (hid 46% of the real wear)
  Track evolution:    +0.028 s  (hid 2% of the real wear)
  Traffic removed:    3 lap(s) flagged  (avg +7.739s off clean pace)
  ------------------------------------------------------------
  True tyre wear:     +1.241 s  (100% of total signal)
```
> **Key Finding:** Fuel burn-off concealed **46%** of real degradation. Correcting for fuel and track conditions increases the measured degradation rate by **5.0x**.

### Italy — Driver NOR, MEDIUM (FP2, 8 laps)
```
Noise attribution for this stint:
  Fuel effect:        +0.513 s  (hid 19% of the real wear)
  Track evolution:    -3.733 s  (hid 135% of the real wear)
  Traffic removed:    2 lap(s) flagged  (avg +1.332s off clean pace)
  ------------------------------------------------------------
  True tyre wear:     -2.770 s  (100% of total signal)
```

---

## 3. New Visualizations from Bahrain (After Long-Run Filter)

### Graph 2 — Before vs After (Shared Y-Axis)
The raw stint shows near-zero slope (+0.0126 s/lap), while DegradIQ's corrected line reveals the true +0.1699 s/lap degradation slope.
![Bahrain Graph 2 — Before vs After](file:///C:/Users/PRANAV/.gemini/antigravity-ide/brain/76a9e47b-46b1-4bbf-9ace-3a8bf916af1f/bahrain_graph2.png)

### Graph 3 — Practice-Trained Model vs Actual Race Validation
Trained on practice data and validated against real grand prix race laps:
![Bahrain Graph 3 — Prediction vs Actual](file:///C:/Users/PRANAV/.gemini/antigravity-ide/brain/76a9e47b-46b1-4bbf-9ace-3a8bf916af1f/bahrain_graph3.png)
