# 📊 Mini Trading Suggestions App

A data-driven trading suggestions app built with **Python** and **Streamlit** that provides simple trading insights using historical stock data and rule-based logic to help guide buy/sell decision-making.

> 🎬 A demo video (`project 1 demo.mp4`) is included in the repository — check it out to see the app in action!

---

## 🚀 Features

- 📈 Fetches and analyzes **historical stock data**
- 💡 Generates **trading suggestions** (Buy / Sell / Hold) based on smart logic
- 🖥️ Clean, interactive **Streamlit** web interface
- 📓 Jupyter Notebook for exploratory data analysis and prototyping

---

## 🗂️ Project Structure

```
mini-trading-App/
├── mini trading app/       # Main application source code
│   ├── app.py              # Streamlit app entry point
│   └── *.ipynb             # Jupyter Notebook(s) for analysis & prototyping
└── project 1 demo.mp4      # Demo video of the app
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Interactive web UI |
| Jupyter Notebook | Data exploration and prototyping |
| pandas | Data manipulation |
| yfinance | Stock data source |

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/SameenYounas1/mini-trading-App.git
cd "mini-trading-App/mini trading app"
```

### 2. Install dependencies

```bash
pip install streamlit pandas yfinance
```

> If a `requirements.txt` is present, use:
> ```bash
> pip install -r requirements.txt
> ```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📓 Jupyter Notebook

The repository includes a Jupyter Notebook used for exploring the data and building out the trading logic before integrating it into the Streamlit app. To open it:

```bash
jupyter notebook
```

---

## 💡 How It Works

1. **Data Retrieval** — Pulls historical stock price data for a given ticker symbol.
2. **Analysis** — Applies rule-based or indicator-driven logic (e.g., moving averages, price trends) to assess the stock's momentum.
3. **Suggestion** — Outputs a trading recommendation based on the analysis.
4. **UI** — Presents the results in a clean, user-friendly Streamlit interface.

---

## ⚠️ Disclaimer

This app is intended for **educational purposes only**. The trading suggestions it provides are based on simplified logic and historical data — they are **not financial advice**. Always do your own research before making any investment decisions.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
