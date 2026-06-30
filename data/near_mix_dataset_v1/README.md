# Near-Mix Validation Dataset v1

## Dataset Purpose

This dataset is generated from AliMeeting near/headset tracks. For each meeting, all participant near microphones are time-aligned and summed into one clear multi-speaker mixture:

- `*_near_all_speakers_mix.wav`: all-speaker close-talk mixture for project separation testing.
- `*_source_XX_N_SPK*.wav`: per-speaker reference sources used to create the mixture.
- `manifest.jsonl`: one JSON record per meeting.
- `summary.json`: dataset-level summary.

Use this dataset to test whether the project can separate clear multi-speaker speech without the extra difficulty of far-field room acoustics.

## Important Boundary

This is not a real far-field microphone recording. It is a close-talk synthetic mixture:

- clearer than far-field audio;
- still multi-speaker and can include overlap;
- suitable for listening tests and objective separation scoring;
- should be reported as `near/headset close-talk mixture`, not as real meeting far-field input.

## Generation Details

- Source during generation: `data/source/Eval_Ali_near`
- Output directory: `data/near_mix_dataset_v1`
- Meeting count: 3
- Total duration: 5477.493 seconds
- Sample rate: 16000 Hz
- Inactive speaker regions were masked using the corresponding near TextGrid files.
- Same-meeting near tracks were trimmed to the shortest track length when tiny length differences existed.
- The original AliMeeting near/far folders were removed after this derived dataset was generated. To reproduce it later, place the raw near package under `data/source/Eval_Ali_near`.

## Reproduce

```powershell
backend\.venv\Scripts\python.exe scripts\create_near_mix_dataset.py --output-dir data\near_mix_dataset_v1 --mask-inactive
```

## Suggested Use

1. Upload one `*_near_all_speakers_mix.wav` file to the frontend.
2. Run the current best separation path.
3. Compare separated tracks by ear against the corresponding `*_source_*.wav` files.
4. For objective scoring, use `manifest.jsonl` to map each mixture to its reference source tracks.
