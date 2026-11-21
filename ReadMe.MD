HumanGestureMenzerathAltman/
├── Code_Processing/
│   ├── FeatureGeneration/                 # Feature processing (not rerun in the notebook)
│   │   ├── tracking.py                    # - Mediapipe holistic tracking script on non-anonymized data
│   │   ├── animation_creation.py          # - anonymized animations of each gesture showing upper body 3D skeleton data
│   │   └── requirements.txt
│   └── PeakExtractionCode/                # Extracts submovements from .npz skeleton files
│       ├── processingGestures_method1.py  # - Peak method 1: speed peaks above 15cm p/s
│       ├── processingGestures_method2.py  # - Peak method 2: speed peaks above 10% of the max speed of the gesture
│       │                                  # - Both remove duplicate files, compute submovement durations, path lengths
│       │                                  # - Both output gestures_flat and gestures_long (.csv)
│       └── requirements.txt
├── Data_Input/                            # Input data (anonymized), CC-BY-NC-SA 4.0 (raw data not included)
│   ├── Gesture/                           # Gesture skeleton files (.npz, .mp4) per corpus
│   └── Objman/                            # Object manipulation files (.npz, .mp4) (ECOLANG only)
├── Data_Output/                           # Processed datasets (one set per peak method)
│   ├── gestures_flat_method1.csv          # One row per gesture
│   ├── gestures_long_method1.csv          # One row per submovement (this is used for modeling)
│   ├── duplicates_removed_method1.csv     # Files that were removed as exact duplicates
│   ├── processing_log_method1.txt         # Output of the script (Ns per corpus)
│   └── ..._method2                        # Same four files for peak method 2
├── animations/                            # Example animations shown in the notebook (re-encoded to H.264)
│   ├── example_timeseries_head.txt        # First rows of one example time series
│   └── visibility_per_dataset.csv         # Average visibility of the index fingers per dataset
├── plots/                                 # Output figures (PNG, 600 dpi), one set per peak method
│   ├── speedpeaks.png                     # Panel A of the main figure (made by hand)
│   └── old_handplots/                     # Earlier figures that were composed by hand
├── docs/                                  # Rendered Quarto website
├── quarto_dependencies/                   # Bibliography files
│   └── references/refs.bib
├── old_scripts/                           # Earlier versions of the analysis (not rendered)
├── feature_processing.qmd                 # 1. Shows tracking code, example animations, example time series
├── preprocessing.qmd                      # 2. Runs the two peak extraction scripts
├── main_analysis_1.qmd                    # 3. Analysis, figure and null model for peak method 1
├── main_analysis_2.qmd                    # 4. Same for peak method 2
├── _analysis_body.qmd                     # The analysis itself, shared by main_analysis_1 and main_analysis_2
├── null_simulation.qmd                    # 5. Check of the null model on simulated data
├── _quarto.yml                            # Quarto configuration (render order of the pages)
└── README.md