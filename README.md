# 🛡️ ScamShield AI

### AI-Powered Scam Detection Platform

ScamShield AI is a machine learning-based security platform designed to identify potential online scams and fraudulent activities. It analyses SMS messages, job postings, and website URLs to help users recognize suspicious content.

The application combines three machine learning detection modules in one interactive Streamlit dashboard.

---

## 🚀 Features

### 1. 📩 SMS Scam Detection
- Detects spam and suspicious SMS messages.
- Uses TF-IDF for text feature extraction.
- Uses Logistic Regression for classification.
- Displays prediction confidence.

### 2. 💼 Fake Job Detection
- Analyses job titles, company profiles, descriptions, requirements, and benefits.
- Identifies potentially fraudulent job postings.
- Uses TF-IDF and Logistic Regression.

### 3. 🔗 Phishing URL Detection
- Analyses website URL characteristics.
- Extracts 17 URL-based features.
- Uses Logistic Regression to classify potentially phishing URLs.

### 4. 📊 Interactive Dashboard
- Displays total analyses.
- Tracks suspicious and normal results.
- Shows recent detection activity.

### 5. 🗄️ Detection History
- Stores detection results using SQLite.
- Preserves history across browser refreshes and application restarts.
- Displays previous detection results.
- Supports clearing saved history.

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Text Processing | TF-IDF |
| Classification | Logistic Regression |
| Frontend | Streamlit |
| Database | SQLite |
| Model Storage | Joblib |

---

## 🧠 Machine Learning Models

| Detection Module | Algorithm | Accuracy |
|---|---|---:|
| SMS Detection | TF-IDF + Logistic Regression | 97.58% |
| Fake Job Detection | TF-IDF + Logistic Regression | Not yet documented |
| URL Detection | Logistic Regression | 99.33% |

*Accuracy values are based on the recorded evaluation results for the SMS and URL models. They are dataset-specific and do not guarantee real-world detection performance.*

---

## 📂 Project Structure

```text
ScamShield-AI/
│
├── app.py
├── database.py
├── scamshield.db
│
├── models/
│   ├── scamshield_model.pkl
│   ├── fake_job_model.pkl
│   └── url_model.pkl
│
├── datasets/
│   ├── SMSSpamCollection
│   └── fake_job_dataset.csv
│
├── venv/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project

```bash
cd ScamShield-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

**Windows:**

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 📈 How It Works

1. The user selects a detection module.
2. The user enters an SMS message, job posting, or website URL.
3. The application extracts the required features.
4. The trained machine learning model analyses the input.
5. The application displays the predicted result and confidence.
6. The detection result is stored in the SQLite database.
7. The dashboard updates with the latest detection activity.

---

## 🔮 Future Improvements

- Improve model performance with additional datasets.
- Add more advanced phishing URL analysis.
- Improve fake job detection.
- Add downloadable detection reports.
- Add user authentication.
- Deploy the application online.

---

## ⚠️ Disclaimer

ScamShield AI is an educational machine learning project. Its predictions are not guaranteed to be accurate and should not be treated as proof that a message, job posting, or website is safe or fraudulent.

Always independently verify suspicious messages, employers, and websites.

---

## 👨‍💻 Developer

**Dhanush (H Dhanush)**

- GitHub: [dhanush080607](https://github.com/dhanush080607)
- LinkedIn: [H Dhanush](https://www.linkedin.com/in/h-dhanush-189565327/)
- Portfolio: [Dhanush Portfolio](https://dhanush-portfolio2.vercel.app/)

---

⭐ If you find this project interesting, consider giving the repository a star!