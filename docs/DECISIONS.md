# Decisions

One line of reasoning each. Numbers cited here are produced by `scripts/profile_data.py` and `scripts/design_target.py`.

1. The shipped file wins over the brief: 8,790 rows and 10 columns, with no `cast` and no `description`. Empty strings fill those columns so later code has a stable frame, and description TF-IDF is not fitted on empty text.
2. `Not Given` in `director` (2,588) and `country` (287) is missing data, not a person or a place, and it is not rewritten to the string Unknown.
3. `date_added` mixes `M/D/YYYY` and ISO dates, so the parser tries an explicit format list and coerces the rest. A single strptime format would drop the other half.
4. `NR` and `UR` are dropped from the ordinal model. They mean "not certified", which is not a step on a maturity ladder.
5. Rare raw classes are folded using counts on the training pool only, after the sealed split, so the test set does not decide the label space. A class still under 5 rows after folding is dropped rather than breaking `StratifiedKFold`.
6. Formulation C is routed by `type`, not by certification system. Most movies in this file carry a TV certificate, so a film-only movie model would discard them. The exact share is in `reports/target_design.md`.
7. The sealed 15% split is stratified on the maturity bucket (unrated rows are their own stratum) and is not scored in the label-design report.
8. Comparable formulation metrics are ordinal metrics on the shared 4-level ladder. Native accuracy is not comparable across label sets of different sizes, and it is always shown next to a most-frequent dummy.
9. Comparison models use the fixed hyperparameters in `config.yaml`. They are not a tuned result. Tuning happens after the product formulation is chosen.
10. `ComplementNB` gets a non-negative preprocessor. Standardised features and a negative `years_to_platform` are not legal inputs for that model.
11. `TV-PG` and `PG` share Older Kids. They are adjacent guidance certificates below the teen line; splitting them would recreate the two-system problem inside one boundary.
12. `NC-17` stays inside Mature. Three titles cannot be their own stratified class, and dropping them would hide the top of the ladder.
13. Genre indicators are a content descriptor, not a restatement of the certificate. Mutual information on the training pool is the check that no single feature, genre included, separates the label.
14. Lexicon counts and `n_directors` are extra numeric features because `cast` is absent and the title is the only text. They are fixed lists and row-local counts, not statistics fit on the data.
15. Title TF-IDF uses `strip_accents="unicode"` so accented and unaccented forms of the same token can match. The brief specified that flag for the description vectorizer; the description column is empty, so the flag is applied to the text we actually have.
16. The product formulation is whichever label space the pre-registered rule in `scripts/design_target.py` selects from logistic-regression pooled QWK, confirmed with random forest. The rule is not "always B".
17. `LOKY_MAX_CPU_COUNT` is set before scikit-learn imports. On this Windows build, joblib's CPU probe calls `wmic` and never returns.
18. Comparison `HistGradientBoosting` uses `max_iter=50` and `max_bins=64`. The default 255-bin histogram on the dense title matrix was still running after 25 minutes for a single formulation, so the comparison budget was cut and recorded here rather than presented as a tuned result.
19. Formulation B is tuned with `HalvingRandomSearchCV` and 40 candidates, capped at 2,000 rows per rung. Each family's winner is then scored with a real 5-fold on the full training pool. A flat 40-by-5 search on all 7,402 rows did not fit a laptop budget.
20. The sealed test set is scored once, after the forest is frozen, by `scripts/evaluate_test.py`. Those scores do not change parameters. The `class_weight=None` forest in the same report is a pre-specified comparison, not a search, and it does not replace the saved model.
21. A public rating API is not used to raise accuracy. This catalogue is public, so a language model can recite the certificate, which would leak the label. The accuracy attempt is a soft vote of the three already-tuned models. The eight weight mixes were written down before the out-of-fold run. A mix replaces the forest only when both out-of-fold accuracy and out-of-fold macro-F1 are higher. That choice does not see the holdout.
22. That vote, a short list of class-probability scales, and a `class_weight=None` forest were all scored on training folds. None beat the balanced forest on both accuracy and macro-F1, so the saved model was not replaced and the holdout was not scored a second time for model choice.
