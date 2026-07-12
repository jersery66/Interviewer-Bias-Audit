# Model configuration for the analysis-locked audit

| Component | Frozen configuration |
|---|---|
| Split | Repeat 1 of the existing 10 x 5 participant-level split file; five outer folds; one cross-fitted test prediction per participant |
| Fold fitting | Vocabulary, TF-IDF weights, numeric scaling, calibration, and threshold selection fit without outer-test participants |
| TF-IDF | lowercase=True; strip_accents=unicode; word ngrams=(1, 2); min_df=2; max_features=50,000; sublinear_tf=True |
| Classifier | LogisticRegression; C=1.0; class_weight=balanced; solver=liblinear; max_iter=2,000 |
| Numeric features | StandardScaler fit on each outer-training fold |
| Random seed | Base seed 20260705; fold/model offsets recorded in source code and run manifest |
| Calibration | Platt and isotonic calibrators fit from inner cross-fitted training predictions, then applied to the outer test fold |
| Default operation | Probability 0.50 is a default operating point, not a clinical or preregistered cutoff |
| Alternative thresholds | Selected only from inner training predictions for max Macro-F1, max Youden J, and sensitivity subject to specificity >=0.80 |
| Uncertainty | 5,000 participant bootstrap replicates; paired tests use 10,000 within-participant swaps |
| Multiplicity | Original five analysis-locked comparisons retain their own BH family; post-audit positional and C5 sensitivity families are adjusted separately |

The analysis-locked split was fixed before the post-audit sensitivity run to prevent further split selection. This is not formal preregistration.
