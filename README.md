# 📊 Demand & Sales Forecasting

This repository contains two implementations of a **Demand & Sales Forecasting** application using **Facebook Prophet** for time-series forecasting.  
Both versions provide the same functionality, but differ in the frontend framework used:

- **Flask** → Web app with server-side rendering (`templates/index.html`)  
- **Streamlit** → Interactive dashboard for quick prototyping and visualization  

---

## 📂 Repository Structure

├── Flask/ # Flask implementation
│ ├── app.py # Main Flask app
│ ├── templates/ # HTML frontend
│ ├── static/ # (Optional) CSS/JS/Assets
│ ├── train.csv # Historical sales dataset
│ ├── places.csv # Store information
│ └── products.csv # Product information
│
├── Streamlit/ # Streamlit implementation
│ ├── app.py # Main Streamlit app
│ ├── Logo.jpg # Project logo
│ └── train.csv # Historical sales dataset
│
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Features

- Forecast future demand using **Prophet**
- Visualize **past sales trends**
- Estimate **stock depletion timeline**
- Suggest **optimal daily imports** to avoid stockouts
- Compare **Actual vs Predicted demand**
- Two frontend options:
  - **Flask** (production-ready web app)
  - **Streamlit** (interactive data app for exploration)

---

## 🛠️ Installation

Clone the repo:

```bash
git clone https://github.com/your-username/demand-forecasting.git
cd demand-forecasting
Install dependencies (preferably in a virtual environment):

bash
Copy code
pip install -r requirements.txt
Example requirements.txt:

nginx
Copy code
Flask
pandas
prophet
streamlit
numpy
streamlit-gchart
🚀 Running the Applications
▶️ Flask App
bash
Copy code
cd Flask
python app.py
Open your browser and go to http://127.0.0.1:5000/

Choose Store, SKU, forecast duration, and stock to see results.

▶️ Streamlit App
bash
Copy code
cd Streamlit
streamlit run app.py
Streamlit will start a local server (default: http://localhost:8501)

Use the sidebar to configure forecast settings and visualize results interactively.

📊 Example Outputs
Past Sales Trends: Yearly sales by store & SKU

Demand Forecast vs Stock: Shows when stock will run out

Optimized Stock Import Plan: Suggests daily import strategy

Actual vs Predicted: Compare historical test data with forecasts

📌 Notes
Both versions use the same datasets (train.csv, places.csv, products.csv)

Flask version exposes APIs (/POST) returning JSON outputs

Streamlit version is UI-driven with inline charts & tables

Prophet model parameters can be tuned for better accuracy

🤝 Contributing
Pull requests and issues are welcome.
If you’d like to enhance the UI, add new forecasting methods, or improve model accuracy, feel free to contribute!