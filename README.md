# Human Activity Recognition — CNN-LSTM

A CNN-LSTM model for 12-class human activity recognition from smartphone accelerometer/gyroscope
data, comparing raw sensor sequences against hand-engineered features to see which representation
actually drives performance. It was built for **CS3244 Machine Learning (NUS)** by a 6-person
team; my personal contribution was the CNN-LSTM section, documented below.

## Group project — CS3244 Machine Learning, Group 32

**Team:** Elysia Selena Alston, Jiyuan Lu, Nabil Rakaiza Abror, Tiara Mirchandani, Wang Kaitian,
Zakariyyaa Chachia

**My contribution (this README's focus):** the CNN-LSTM framework in full — architecture design, the
raw-sequence vs. engineered-feature comparison, and the class-imbalance experiments (class weighting
vs. SMOTE). The other model families in this repo (logistic regression, SVM, 1D-CNN) were built by
teammates as part of the shared group project; this README documents my part specifically.

## Task & dataset

12-class activity classification (6 static/dynamic activities + 6 postural transitions, e.g.
`SIT_TO_STAND`, `STAND_TO_LIE`) from the Recognition of Human Activities Using Smartphones dataset —
30 users, 561 engineered features from raw accelerometer/gyroscope signals, >10,000 samples, with
severe class imbalance on the transition classes (some under 25 samples).

## What I built

Two CNN-LSTM variants, trained and evaluated under matched conditions:

- **Raw-sequence model** — deeper 3×Conv1D block → 2-layer LSTM (128 units), fed 128-timestep
  sliding windows (50% overlap) of raw sensor data. 312K parameters.
- **Engineered-feature model** — 2×Conv1D block (64 filters) → 2-layer LSTM (64 units), fed the
  561-dim pre-computed feature vectors. 84K parameters.

Both were evaluated with and without class-weighting and SMOTE, and with a strict subject-wise
train/test split (users 1–26 train, 27–30 test) to avoid leakage.

## Results

| Input | Accuracy | Macro F1 | MIoU | Params |
|---|---|---|---|---|
| Raw sequences (baseline, no rebalancing) | 94.24% | 0.8685 | 0.7929 | 312,012 |
| Engineered features (baseline, no rebalancing) | 85.36% | 0.54 | 0.4615 | 84,108 |

The same architecture went from the weakest model in the study to the strongest depending only on
its input representation — raw sequences preserved the temporal ordering (e.g. distinguishing
`SIT_TO_LIE` from `LIE_TO_SIT`) that the engineered features had already discarded.

Class weighting and SMOTE both slightly *hurt* the raw-data model's macro F1 relative to the
unweighted baseline — the architecture already learned robust representations from the real
minority-class samples, so external rebalancing added noise rather than signal.

**Error analysis:** the raw-data model's main confusion was an asymmetric SITTING↔STANDING mix-up —
a 9.60% pairwise misclassification rate, with actual `SITTING` misclassified as `STANDING` (14.93%)
over three times more often than the reverse (4.58%), pointing to overlapping feature representations
in the data itself rather than a model-capacity issue.

## Notes

- The subject-wise train/test split (rather than a random window-level split) is what keeps the
  reported numbers honest here — it's easy to get inflated accuracy on this dataset if windows from
  the same subject leak across train and test.
