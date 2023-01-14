## Deployment

* Model filepaths:
    * `exp/{voice}/tts_train_raw_char`
    * `exp/{voice}/tts_stats_raw_char`

* Vocoder filepath:
    * `exp/{voice}/vocoder`

* Change in `exp/{voice}/tts_train_raw_char/config.yaml`:

  `stats_file: exp/tts_stats_raw_char/train/feats_stats.npz` to
  `stats_file: exp/{voice}/tts_stats_raw_char/train/feats_stats.npz`

* In `synthesize.py`:

  set voices = [] accordingly