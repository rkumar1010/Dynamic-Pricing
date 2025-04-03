# Dynamic Pricing Machine Learning Model

This project implements a **Machine Learning model for Dynamic Pricing**, using a **Random Forest Regressor** to predict optimal prices based on demand, competition pricing, seasonality, and base price.

## Features
- **Synthetic Data Generation**: Creates a dataset simulating real-world pricing factors.
- **Data Preprocessing**: Scaling and splitting for efficient model training.
- **Machine Learning Model**: Uses **Random Forest Regressor** for price prediction.
- **Evaluation**: Measures performance using **MAE and RMSE**.
- **Visualization**: Compares actual vs. predicted prices.

## Installation

Clone this repository:
```bash
git clone https://github.com/rkumar1010/dynamic-pricing.git
cd dynamic-pricing
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter Notebook:
```bash
jupyter notebook dynamic_pricing.ipynb
```

## Project Structure
```
├── dynamic_pricing.ipynb  # Main Jupyter Notebook
├── requirements.txt       # Dependencies
├── README.md              # Project Documentation
├── data/                  # (Optional) Store dataset
└── models/                # (Optional) Store trained models
```

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

To install them manually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## License
This project is open-source and available under the **MIT License**.

## Contributing
Feel free to fork the repo, open issues, or submit pull requests!
