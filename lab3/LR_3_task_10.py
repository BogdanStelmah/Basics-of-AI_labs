import datetime
import json
import numpy as np
from sklearn import covariance, cluster
import yfinance as yf

# Вхідний файл із символічними позначеннями компаній
input_file = "company_symbol_mapping.json"

# Завантаження прив'язок символів компаній до їх повних назв
with open(input_file, "r") as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Визначення архівних даних котирувань
start_date = "2003-07-03"
end_date = "2007-05-04"

# Завантаження архівних даних котирувань
quotes = []
valid_symbols = []
for symbol in symbols:
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if not data.empty:
            quotes.append(data)
            valid_symbols.append(symbol)
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")

# Перевірка чи є валідні дані
if not quotes:
    print("No valid data available for any symbol. Check your symbol mapping and data availability.")
else:
    # Оновлення символів на дійсні
    symbols = valid_symbols

    # Вилучення котирувань, що відповідають відкриттю та закриттю біржі
    opening_quotes = np.array([quote["Open"].values for quote in quotes]).T
    closing_quotes = np.array([quote["Close"].values for quote in quotes]).T

    # Обчислення різниці між двома видами котирувань
    quotes_diff = closing_quotes - opening_quotes

    # Нормалізація даних
    X = quotes_diff.copy()
    X /= X.std(axis=0)

    # Створення моделі графа
    edge_model = covariance.GraphicalLassoCV()

    # Навчання моделі
    with np.errstate(invalid="ignore"):
        edge_model.fit(X)

    # Створення моделі кластеризації на основі поширення подібності
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    num_labels = labels.max()

    # Виведення результатів
    print("\nClustering of stocks based on difference in opening and closing quotes:\n")
    for i in range(num_labels + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_names = names[cluster_indices]
        if len(cluster_names) > 0:
            print("Cluster", i + 1, "==>", ", ".join(cluster_names))
