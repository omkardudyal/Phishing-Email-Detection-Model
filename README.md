# Thiranex Phishing Email Detection Model

A beginner-friendly machine-learning project that classifies email text as **Phishing** or **Safe** using Scikit-learn.

## Features
- Trains on labeled phishing and legitimate/safe email examples.
- Uses TF-IDF text features and simple URL/text warning indicators.
- Reports accuracy, precision, recall, F1-score, and a confusion matrix.
- Lets you enter an email subject and body to get a prediction.

## Requirements
- Python 3.10+ recommended
- Internet connection for installing packages (only needed once)

## Run on Windows
Open Command Prompt in this project folder and run:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

If `py` is not recognized, use `python` instead of `py` in the first command.

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The program prints evaluation metrics and saves `confusion_matrix.png` in the project folder. Then enter a subject and body to test an email. Type `exit` to stop.

## Dataset
`data/emails.csv` is a small, balanced starter dataset written for this educational demonstration. It is intentionally small and **must not be presented as a large real-world benchmark**. For a stronger project, replace it with a properly licensed public email dataset after confirming its column names and labels, then update the loading section in `main.py` if needed.

Each row contains:
- `subject`: email subject
- `body`: email body
- `label`: `Phishing` or `Safe`

## Limitations
- The sample dataset is small and examples are simplified; the displayed accuracy may not generalize to real emails.
- The confidence score is not proof that an email is safe.
- Never click suspicious links or share passwords, bank details, or OTPs based on a model prediction.

## Project structure
```text
thiranex_phishing_email_detector/
├── data/
│   └── emails.csv
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Suggested future improvements
- Train on a larger, trustworthy email dataset.
- Add a simple Streamlit web interface.
- Compare Logistic Regression with Naive Bayes and Random Forest.
- Add tests and report per-class recall.

## Credits
Built as a student project for the Thiranex Phishing Email Detection Model task. Uses pandas, SciPy, Matplotlib, and Scikit-learn.
