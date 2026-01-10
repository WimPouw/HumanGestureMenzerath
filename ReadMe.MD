HumanGestureMenzerathAltman/
```text
├── Code_Processing/          # Python scripts for processing skeleton data
├── Data_Input/               # Input data (anonymized)
│   ├── Gesture/              # Gesture skeleton files (.npz, .mp4) per corpus
│   └── Objman/               # Object manipulation files (.npz, .mp4) (ECOLANG only)
├── Data_Output/              # Processed datasets
│   ├── gestures_flat.csv     # One row per gesture
│   └── gestures_long.csv     # One row per submovement (this is used for modeling)
├── plots/                    # Output figures (SVG/PNG)
├── docs/                     # Rendered Quarto website
├── quarto_dependencies/      # Bibliography files
│   └── references/refs.bib
├── main_analysis_1.qmd       # Main analysis R (Quarto document) -> THIS IS RUN IN RMARKDOWN
├── _quarto.yml               # Quarto configuration
└── README.md
```

Notebook: https://wimpouw.github.io/HumanGestureMenzerathAltman/main_analysis_1.html

contact: w.pouw@tilburguniversity.edu, kadava@leibniz-zas.de, fuchs@leibniz-zas.de