# LifeCycleProject

## Predicting Next Day’s Close Movement (Positive / Negative)

---

## Project Structure

* **`/data`**
  Contains the raw datasets used throughout the project.

* **`/notebooks`**
  Contains well-organized Jupyter notebooks covering:

  * Data manipulation
  * Exploratory data analysis
  * Feature engineering
  * Model testing & evaluation

* **`/src`**
  Contains modular source code split into subfolders:

  * **models**

    * Model evaluators that provide both metrics and visualization
    * Base model blueprints defining standard behaviors
  * **main**

    * Core scripts responsible for model training
  * **preprocessing_pipeline**

    * Consolidated data preprocessing steps extracted from notebooks, organized into a reusable pipeline

---

## Dataset Selection and Rationale

We selected the **B3 Historical Quotes** dataset, which contains comprehensive historical price data from the **Brazilian stock exchange (B3)**. This real-world financial dataset allows us to build a practical understanding of market behavior and predictive modeling in finance.

### Key Dataset Components:

1. **Top 5 / Top 30 Companies by Data Volume**
   These stocks are the most actively traded and provide rich data for detecting trading patterns and price movement dynamics.

2. **Market-Representative Funds (Ibovespa Index)**
   The Ibovespa is the benchmark index of the Brazilian stock market. Including this index and its tracking funds enables us to incorporate broader market trends and macroeconomic factors into our models.

---

## Getting Started

