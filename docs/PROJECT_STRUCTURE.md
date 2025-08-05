# Project Structure

## Current Organized Directory Structure

```
GNN-FraudDetection/
├── README.md                    # Main project documentation
├── CHANGELOG.md                 # Version history and changes
├── FEATURES.md                  # Feature documentation
├── LICENSE                      # Project license
├── requirements.txt             # Python dependencies
├── config.yaml                  # System configuration
├── .gitignore                   # Git exclusions
│
├── main.py                      # Main entry point
├── demo.py                      # System demonstration script
├── diagnostic.py                # System diagnostic tool
├── test_system.py               # System tests
│
├── setup_environment.sh         # Linux/Mac setup script
├── setup_environment.bat        # Windows setup script
│
├── src/                         # Source code modules
│   ├── data/                    # Data processing modules
│   ├── models/                  # GNN model implementations
│   ├── training/                # Training utilities
│   ├── utils/                   # Utility functions
│   ├── evaluation/              # Model evaluation
│   └── visualization/           # Data visualization
│
├── data/                        # Data storage (excluded from Git)
│   ├── raw/                     # Raw dataset files (.gitkeep only)
│   ├── processed/               # Processed data (.gitkeep only)
│   └── graphs/                  # Graph structures (.gitkeep only)
│
├── results/                     # Model outputs (excluded from Git)
│   └── models/                  # Trained models (.gitkeep only)
│       ├── checkpoints/         # Training checkpoints
│       └── trained/             # Final trained models
│
├── logs/                        # System logs (excluded from Git)
│   ├── training/                # Training logs
│   └── inference/               # Inference logs
│
├── webapp/                      # Streamlit dashboard
│   └── app.py                   # Dashboard application
│
├── docs/                        # Documentation
│   ├── architecture.md          # System architecture
│   ├── quick_start.md           # Quick start guide
│   ├── TROUBLESHOOTING.md       # Issue resolution guide
│   ├── GITIGNORE_SETUP.md       # Git configuration details
│   └── SETUP_SUMMARY.md         # Setup summary
│
└── .venv/                       # Virtual environment (excluded from Git)
```

## Key Organization Principles

### 1. **Clean Root Directory**
- Only essential files in root
- Scripts organized by purpose
- Clear naming conventions

### 2. **Logical Module Structure**
- `src/` contains all source code
- Modules organized by functionality
- Clear separation of concerns

### 3. **Data Management**
- All data files excluded from Git
- Directory structure preserved with .gitkeep
- Clear separation of raw, processed, and graph data

### 4. **Documentation Organization**
- Main docs in `docs/` directory
- README.md in root for quick access
- Specialized guides for different needs

### 5. **Development Support**
- Automated setup scripts for different platforms
- Comprehensive diagnostic tools
- Clear testing structure

## Removed Items

### Redundant Files Removed:
- `install.py` → Replaced by `setup_environment.sh/bat`
- `setup.py` → Renamed to `demo.py` for clarity
- `__pycache__/` → Removed Python cache
- `.mypy_cache/` → Removed type checker cache

### Cache and Temporary Files:
- All `.pyc` files removed
- Cache directories cleaned
- Temporary files purged

## Benefits of Current Organization

1. **Clear Purpose**: Each directory and file has a specific, obvious purpose
2. **Scalability**: Structure supports project growth
3. **Maintainability**: Easy to find and modify components
4. **Collaboration**: Standard structure familiar to developers
5. **Git Efficiency**: Large files excluded, only code tracked
6. **Documentation**: Comprehensive guides for all aspects

## Navigation Guide

### For Users:
- Start with `README.md`
- Run `./setup_environment.sh` (Linux/Mac) or `setup_environment.bat` (Windows)
- Use `python diagnostic.py` to verify setup
- Follow `docs/quick_start.md` for usage

### For Developers:
- Source code in `src/`
- Tests in `test_system.py`
- Configuration in `config.yaml`
- Architecture details in `docs/architecture.md`

### For Data Scientists:
- Data processing in `src/data/`
- Models in `src/models/`
- Evaluation in `src/evaluation/`
- Visualization in `src/visualization/`

This organization provides a professional, scalable, and maintainable project structure suitable for both development and production use.
