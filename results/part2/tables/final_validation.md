### Final classifier (all 7 generators): validation split, in-distribution only, not generalisation

| backbone | pipeline | format | auroc | ap | balanced_accuracy | fake_recall | real_specificity | n_real | n_fake |
|---|---|---|---|---|---|---|---|---|---|
| clip_l14 | A | matched_jpeg | 0.9817 | 0.9851 | 0.9323 | 0.9289 | 0.9358 | 2961 | 3500 |
| clip_l14 | A | raw | 0.9865 | 0.9891 | 0.9433 | 0.9389 | 0.9477 | 2961 | 3500 |
| clip_l14 | B | matched_jpeg | 0.9800 | 0.9815 | 0.9276 | 0.9366 | 0.9186 | 2961 | 3500 |
| clip_l14 | B | raw | 0.9955 | 0.9960 | 0.9697 | 0.9751 | 0.9642 | 2961 | 3500 |
