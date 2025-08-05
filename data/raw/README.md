# IEEE-CIS Fraud Detection Dataset

This directory should contain the IEEE-CIS Fraud Detection dataset from Kaggle.

## Required Files

The following files should be downloaded and placed in this directory:

1. **train_transaction.csv** (~500MB)
   - Main training dataset with transaction information
   - Contains 590,540 training samples
   - Features: TransactionDT, TransactionAMT, ProductCD, card1-card6, addr1-addr2, dist1-dist2, etc.

2. **train_identity.csv** (~30MB)
   - Identity information for training transactions
   - Contains additional identity verification data
   - Features: id_01-id_38, DeviceType, DeviceInfo

3. **test_transaction.csv** (~400MB)
   - Test dataset for predictions
   - Contains 506,691 test samples
   - Same structure as train_transaction.csv but without fraud labels

4. **test_identity.csv** (~25MB)
   - Identity information for test transactions
   - Same structure as train_identity.csv

5. **sample_submission.csv** (~15MB)
   - Sample submission format for Kaggle competition
   - Shows required output format

## Download Instructions

### Option 1: Kaggle CLI (Recommended)
```bash
pip install kaggle
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip
mv *.csv /path/to/project/data/raw/
```

### Option 2: Manual Download
1. Go to: https://www.kaggle.com/c/ieee-fraud-detection/data
2. Download all CSV files
3. Place them in this directory

## Data Verification

After downloading, verify files are present:
```bash
ls -la data/raw/
python diagnostic.py  # Run system diagnostic
```

## Dataset Information

- **Total Size**: ~1GB
- **Competition**: IEEE-CIS Fraud Detection
- **Problem Type**: Binary classification (fraud/legitimate)
- **Evaluation Metric**: Area Under the ROC Curve (AUC-ROC)
- **Time Period**: Transactions from real-world e-commerce platform

## Important Notes

- Data files are too large for git repository
- Files are automatically ignored by .gitignore
- Download requires Kaggle account (free)
- Stable internet connection recommended due to file sizes
