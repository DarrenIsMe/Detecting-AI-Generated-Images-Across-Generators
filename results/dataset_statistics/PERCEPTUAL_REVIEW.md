# Perceptual duplicate review

The full audit used 64-bit perceptual hashes with Hamming distance <= 4 to identify review candidates. These candidates are advisory only and are not used as split-group constraints unless manually confirmed.

Reviewed on 4 November 2026 using `duplicate_review.png`:

| Candidate filenames | Decision | Reason |
|---|---|---|
| `073_biggan_00141.png`, `522_sdv5_00195.png` | False positive | Dark spider image versus white ball on grass |
| `VQDM_1000_200_06_633_vqdm_00106.png`, `n04008634_1747.JPEG`, `455_biggan_00039.png` | False positive | Jewellery, rocket, and button-like image |
| `n02879718_29272.JPEG`, `n03658185_12873.JPEG` | False positive | Bow versus rectangular object |
| `GLIDE_1000_200_00_035_glide_00132.png`, `n03793489_5430.JPEG` | False positive | Turtle versus mouse/computer mouse |
| `VQDM_1000_200_00_073_vqdm_00173.png`, `470_biggan_00186.png` | False positive | Spider versus candle |

Confirmed exact duplicates:

| Filenames | Split relationship | Action |
|---|---|---|
| `n02109961_185.JPEG`, `n02110185_129.JPEG` | Both supplied training | Kept in one atomic duplicate group |
| `n12985857_2116.JPEG`, `ILSVRC2012_val_00000200.JPEG` | Supplied train/validation | Kept in one atomic duplicate group and assigned to only one generated split role |

Unresolved risk: perceptual hashing cannot guarantee detection of all transformed duplicates. Original fake-to-real pair identifiers are not available in Tiny-GenImage.
