```
yuv-colour-space/
├─ data/
│  ├─ real/                         # human‑swatched photos
│  │  └─ .6X/
│  │     ├─ 6X_001.png
│  │     └─ 6X_002.png
│  ├─ recipes/                      # what to render (virtual)
│  │  └─ recipes_v01.parquet
│  ├─ stats/                        # colour distributions per shade (from real)
│  │  └─ shade_stats_v01.h5
│  └─ renders/
│     ├─ v01/                       # version tag
│     │  └─ .6XY/
│     │     ├─ 6XY_0001.png         # tone‑mapped, 16‑bit PNG for ML
│     │     └─ 6XY_0001.exr         # linear EXR (raw)
├─ blender/
│  ├─ env.hdr.exr                   # supervisor’s EXR (HDRI)
│  └─ render_batch.py               # bpy automation script
├─ utils/
│  ├─ mix.py                        # LAB‑space mixing + RGB conversion
│  └─ fit_stats.py                  # fit skew‑normal / GMM per shade
└─ META.md                          # what this version means
```