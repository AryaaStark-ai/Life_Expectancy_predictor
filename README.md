
```markdown
# 🌍 Life Expectancy Prediction App

A web application that predicts life expectancy based on demographic and health-related factors using machine learning.


---

## 🚀 Features

- Predicts life expectancy using user-provided inputs
- Intuitive and responsive form layout
- Tooltips (`ℹ️`) for field descriptions
- Clean and modern UI with a two-column input grid
- Displays predictions in a well-styled result table

---

## 🧠 Machine Learning Model

The app uses a pre-trained machine learning model trained on WHO and UN datasets containing:

- Demographic data
- Healthcare statistics
- Socioeconomic indicators

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, Jinja2 (Flask templates)
- **Backend:** Python, Flask
- **ML Model:** Pickle, Scikit-learn
- **Styling:** Custom CSS (Responsive Grid)

---

## 📦 Project Structure

```
life-expectancy-prediction/
│
├── static/
│   └── styles.css               # Custom CSS styles
│
├── templates/
│   └── index.html               # Main web interface
│
├── model/
│   └── life_expectancy_model.pkl # Trained ML model
│
├── app.py                       # Flask backend
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

---

## 🖥️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/life-expectancy-prediction.git
cd life-expectancy-prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser to view the app.

---

## 📸 Screenshots

### Form Input Page

### Prediction Output

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit pull requests.

---

## 📬 Contact

For feedback, bugs or suggestions:  
**Email:** dholearyaa@gmail.com
