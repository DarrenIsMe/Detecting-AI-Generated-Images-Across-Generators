### Clean condition on the Part 3 sample vs Part 2's published results (reproduction check)

| pipeline   | format       |   auroc_part3_sample |   auroc_part2_full_test |   balanced_accuracy_part3_sample |   balanced_accuracy_part2_full_test |   auroc_difference |
|:-----------|:-------------|---------------------:|------------------------:|---------------------------------:|------------------------------------:|-------------------:|
| A          | matched_jpeg |               0.9528 |                  0.9553 |                           0.8747 |                              0.8767 |            -0.0025 |
| A          | raw          |               0.9659 |                  0.9645 |                           0.8914 |                              0.8903 |             0.0015 |
| B          | matched_jpeg |               0.9642 |                  0.9671 |                           0.8966 |                              0.8997 |            -0.0029 |
| B          | raw          |               0.9888 |                  0.9893 |                           0.9423 |                              0.9451 |            -0.0005 |
