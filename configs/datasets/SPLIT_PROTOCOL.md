# Fixed public-benchmark partitions

All detector variants in Table 14 of the paper use the same immutable dataset partitions. Image
files and their converted annotations always remain in the same subset. Annotation conversion
does not alter the split membership.

| Benchmark | Training | Validation | Evaluation |
| --- | --- | --- | --- |
| DTU/NordTank | released 70% subset | released 15% subset | released 15% test subset |
| WTBD | `train` entries in the provider's `train_val_test_split.txt` | `val` entries in the same index | `test` entries in the same index |
| NEU-DET | 90% of the released 1,440-image training set | class-stratified 10% of the released training set, seed 42 | released 360-image test set |
| DAGM 2007 | 90% of each released Train partition | 10% of each released Train partition, seed 42 | released Test partitions |
| VisDrone-2019 | official 6,471-image train set | official 548-image validation set | official 1,610-image test-dev set |

For the two derived validation subsets, files are first sorted by normalized relative path and
then split with `sklearn.model_selection.train_test_split(test_size=0.10, random_state=42)`. NEU-DET
uses the six defect classes as the stratification labels. DAGM applies the split independently
within each texture class before the subsets are concatenated. The resulting membership is reused
for every model and every training run.

The source datasets do not share a single native annotation type. The paper's common detection
protocol converts the available spatial labels to horizontal bounding boxes. It does not infer
polygon masks, crack skeletons, or curvature fields from those boxes or from weak spatial labels;
the morphology-dependent terms of CA-Shape-IoU are therefore disabled on all five benchmarks.
