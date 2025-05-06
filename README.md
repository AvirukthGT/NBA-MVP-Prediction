# 🏀 NBA MVP Prediction Project

This project aims to predict the NBA Most Valuable Player (MVP) using player and team statistics from [Basketball-Reference](https://www.basketball-reference.com/). The workflow consists of web scraping, data cleaning, feature engineering, and predictive modeling using regression techniques.

---

## Project Structure

| File / Folder           | Description                                               |
|-------------------------|-----------------------------------------------------------|
| `web_scraping.py`       | Collects MVP voting, player stats, and team standings     |
| `Data_cleaning.py`      | Cleans and merges data from multiple sources              |
| `prediction.py`         | Builds regression models and evaluates prediction accuracy|
| `mvps.csv`              | MVP voting data                                           |
| `players.csv`           | Player statistics                                         |
| `teams.csv`             | Team win/loss standings                                   |
| `player_mvp_stats.csv`  | Final merged dataset used for modeling                    |

---

## Overview

The goal is to identify and predict the top MVP candidates each season by analyzing:

- Player performance statistics
- Team success metrics
- Historical MVP voting shares

---

## Steps Involved

### 1. Web Scraping (`web_scraping.py`)
- Scraped MVP voting results, player per-game stats, and team standings from 1991–2024.
- Saved raw HTML pages and parsed tables using `BeautifulSoup` and `pandas`.

### 2. Data Cleaning & Integration (`Data_cleaning.py`)
- Merged MVP, player, and team datasets.
- Removed duplicate team records for players traded mid-season (e.g., using `2TM` or `3TM` rows).
- Replaced team abbreviations with full names using a mapping dictionary.
- Cleaned inconsistencies and handled missing values.
- Final dataset saved as `player_mvp_stats.csv`.

### 3. Prediction & Evaluation (`prediction.py`)
- Used **Ridge Regression** and **Random Forest Regressor** to predict MVP vote shares.
- Defined `predictors` including player performance metrics and normalized stats.
- Validated the model using **year-wise backtesting** (leave-one-year-out).
- Calculated **Average Precision (AP)** to evaluate how well the top 5 predictions match real MVP outcomes.

---

## Features Used

Key predictors include:
- Points (PTS), Assists (AST), Rebounds (TRB), Steals (STL), Blocks (BLK)
- Shooting percentages (FG%, 3P%, FT%)
- Games played (G), Minutes (MP), Turnovers (TOV)
- Team wins/losses and `SRS` (Simple Rating System)
- Normalized year-by-year stat ratios (e.g., `PTS_T`, `AST_R`)

---

## Models Compared

| Model                 | Description                         | Avg Precision (Top-5) |
|----------------------|-------------------------------------|------------------------|
| Ridge Regression     | Linear model with L2 regularization | Moderate               |
| Random Forest        | Ensemble of regression trees        | **Improved**           |

---


To evaluate how well the model ranks the top MVP candidates, I implemented a custom metric called **Average Precision at Top-5 (AP@5)**.

Unlike standard metrics like RMSE or R², this metric specifically measures how many of the **true top 5 MVPs (by voting share)** are ranked highly by the model's predictions. It is particularly useful when the goal is **ranking** rather than exact value prediction.

---
## Custom Loss Function(Named it AP@5)
### How It Works

1. Sort players by actual MVP share and extract the **true top 5**.
2. Sort players by predicted share.
3. As I iterate through the predicted list, I check whether the player is in the actual top 5.
4. Each time a true top-5 player is found, I compute the **precision at that rank**.
5. The final score is the **average of these precision values**.

A perfect score of **1.0** means all top 5 actual MVPs were perfectly ranked at the top by the model.

---

### Formula (Simplified)

$$
\text{AP} = \frac{1}{k} \sum_{i=1}^{n} \frac{\text{Correct Hits}}{\text{Rank}_i} \quad \text{(until k hits found)}
$$

Where:

- $k = 5$ (top-5 MVPs)  
- $\text{Rank}_i$ = position in predicted ranking  
- "Correct Hits" = number of true top-5 players found so far  

---

```python
def find_ap(combination):
    actual=combination.sort_values("Share",ascending=False).head(5)
    predicted=combination.sort_values("predictions",ascending=False)
    ps=[]
    found=0
    seen=1

    for index,row in predicted.iterrows():
        if row["Player"] in actual["Player"].values:
            found+=1
            ps.append(found/seen)
        seen+=1
    return sum(ps)/len(ps)

```
## Time Series Evaluation: Backtesting with `backtest()`

To ensure realistic and forward-looking MVP predictions, I used a **time series backtesting approach** rather than random cross-validation. This mimics how future MVP races would be predicted using only past data, respecting the chronological order of NBA seasons.

### How It Works

The `backtest()` function loops through each season (e.g., 1996 to 2024), and for every year:

1. Trains the model on **all seasons prior to that year**
2. Predicts MVP shares for players in the current year
3. Evaluates the prediction using the custom **AP@5** metric
4. Repeats for each year in the evaluation range

This simulates how the model would have performed if it were deployed at the end of each regular season historically.

---

### Example: Backtest Ridge Regression
```python
def backtest(stats, model, years, predictors):
    aps = []
    all_predictions = []

    for year in years:
        train = stats[stats["Year"] < year]
        test = stats[stats["Year"] == year]

        model.fit(train[predictors], train["Share"])
        predictions = model.predict(test[predictors])

        combination = pd.concat([test[["Player", "Share"]], 
                                 pd.DataFrame(predictions, columns=["predictions"], index=test.index)], axis=1)
        combination = add_ranks(combination)

        aps.append(average_precision_at_5(combination))
        all_predictions.append(combination)

    return sum(aps) / len(aps), aps, pd.concat(all_predictions)
```

```python
from sklearn.linear_model import Ridge
reg = Ridge(alpha=0.1)

mean_ap, aps, all_predictions = backtest(stats, reg, years[5:], predictors)
print("Mean AP@5:", round(mean_ap, 3))
```


## Final Output

- Ranked players by both actual MVP voting share and predicted values.
- Evaluated model performance using difference in actual vs predicted ranks (`Diff`).
- Top players generally aligned well with true MVP finalists.

---

## Results

```text
Top 5 MVP Predictions for 2024:

1. Nikola Jokic
2. Luka Doncic
3. Joel Embiid
4. Jayson Tatum
5. Giannis Antetokounmpo
```
## Future Improvements
- Add advanced stats like PER, BPM, WS
- Use classification models to directly predict MVP
- Integrate a Streamlit dashboard for interactivity

## Acknowledgements
- Data Source: Basketball-Reference
- Libraries: pandas, sklearn, numpy, BeautifulSoup, selenium

🧑‍💻 Author
Created by Avirukth – Master of Data Science, Monash University


