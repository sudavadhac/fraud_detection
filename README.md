# Fraud Detection — MLOps Project

**Author: Suresh D R | AI Product Developer & Technology Mentor**
*MLOps Syllabus — Deploy and Retrain ML Models on AWS*

## Quick Start

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
cd src && python generate_data.py && python train.py && cd ..
streamlit run src/app.py
```

## Project Structure

```
fraud-detection/
├── src/
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── app.py
├── tests/
│   └── test_model.py
├── models/        (gitignored)
├── data/          (gitignored)
├── requirements.txt
└── README.md
```
