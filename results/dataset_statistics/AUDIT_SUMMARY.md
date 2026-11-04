# Tiny-GenImage audit summary

- Revision: `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69`
- Rows audited: 35,000
- Readable images: 35,000
- Corrupt/unreadable: 0
- Duplicate candidate groups: 7
- Exact duplicate groups: 2
- Perceptual candidates requiring review: 5
- Cross-split duplicate groups: 3
- Cross-generator duplicate groups: 4
- Cross-label duplicate groups: 2

## Counts

| Source split | Label | Generator | Count |
|---|---|---|---:|
| train | fake | ADM | 2,000 |
| train | fake | BigGAN | 2,000 |
| train | fake | GLIDE | 2,000 |
| train | fake | Midjourney | 2,000 |
| train | fake | SD15 | 2,000 |
| train | fake | VQDM | 2,000 |
| train | fake | Wukong | 2,000 |
| train | real | Real | 14,000 |
| validation | fake | ADM | 500 |
| validation | fake | BigGAN | 500 |
| validation | fake | GLIDE | 500 |
| validation | fake | Midjourney | 500 |
| validation | fake | SD15 | 500 |
| validation | fake | VQDM | 500 |
| validation | fake | Wukong | 500 |
| validation | real | Real | 3,500 |

## Known limitations

- Tiny-GenImage does not retain an explicit fake-to-real source pairing.
- ImageNet class identifiers are names for real images but numeric indices for generated images; no local mapping joins them.
- Perceptual-hash groups are candidates and may contain false positives; review cross-split groups manually.
- Midjourney is treated as proprietary/unknown architecture, not as confirmed diffusion.
- SD14 exists in the label vocabulary but has zero rows.
