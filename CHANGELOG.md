# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-08-05

### Major Changes
- **Complete Translation**: Translated entire codebase from Portuguese to English
- **Code Organization**: Restructured project following industry best practices
- **Documentation Overhaul**: Professional README with business narrative
- **Feature Documentation**: Comprehensive feature list in FEATURES.md

### Added
- Professional README.md with business problem analysis and technical justification
- FEATURES.md with detailed capability documentation
- Comprehensive project documentation in English
- Enhanced configuration management
- Improved code organization and structure

### Changed
- **Language**: All code comments, docstrings, and documentation translated to English
- **Architecture Documentation**: Updated docs/architecture.md to English
- **Quick Start Guide**: Translated docs/quick_start.md to English
- **Dashboard Interface**: Updated webapp/app.py interface language
- **Configuration**: Updated config.yaml with English comments and documentation

### Fixed
- Removed emoji characters from configuration and code comments
- Standardized variable naming conventions
- Improved code readability and maintainability
- Fixed potential import issues and dependencies

### Technical Improvements
- **Model Architecture**: Maintained all three GNN implementations (GraphSAGE, GAT, HeteroGNN)
- **Data Pipeline**: Preserved complete data processing and graph construction pipeline
- **Training System**: Kept advanced training features including focal loss and early stopping
- **Visualization**: Maintained interactive dashboard and monitoring capabilities
- **Performance**: No performance degradation from translation changes

### Business Value
- **Professional Presentation**: Enterprise-ready documentation and code organization
- **Global Accessibility**: English language ensures broader accessibility
- **Maintainability**: Improved code organization for easier maintenance
- **Compliance**: Professional documentation supports regulatory requirements

### Migration Notes
- All functionality preserved during translation
- Configuration files remain backward compatible
- Model checkpoints and saved models remain valid
- No breaking changes to API interfaces

## [1.0.0] - Previous Version

### Original Features
- GraphSAGE, GAT, and HeteroGNN implementations
- Heterogeneous graph construction from IEEE-CIS fraud detection data
- Advanced feature engineering pipeline
- Real-time fraud detection capabilities
- Interactive Streamlit dashboard
- Comprehensive model evaluation and monitoring
- Production-ready MLOps pipeline
