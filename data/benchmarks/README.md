# Benchmark scenes

`python scripts/prepare_benchmarks.py ...` writes `<dataset>.npz` files here (`data` = H x W x C uint16,
`gt` = H x W uint8, 0 = unlabeled).

| dataset | source | shape | classes |
|---|---|---|---|
| `indian_pines` | AVIRIS Indian Pines (corrected, 200 bands) from the [GIC Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) page; zero-padded to 149 x 149 by `prepare_benchmarks.py` | 149 x 149 x 200 | 16 |
| `pavia_university` | ROSIS Pavia University (same page) | 610 x 340 x 103 | 9 |
| `yu_paint` | in-house YU Paint scene (painted aluminum plates on grass, SPECIM camera, 258 bands) — attached to the GitHub release as `yu_paint_data2_gt2.zip` | 343 x 688 x 258 | 6 |

Indian Pines and Pavia University are third-party data and are not redistributed here. The train/val/test splits in
`data/splits/` index exactly these layouts (10 % / 10 % / 80 % of the labeled pixels per class, seeds 0-4).
