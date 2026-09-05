# Lesson 7: Build Your Next Project ALONE 🚀

You now own every piece of ONE project. Here's the secret:
**every ML project is the same 6 steps with different data.**

DarPrice = houses + prices.
RentalPrice = apartments + rent. CarPrice = cars + price. SalaryPredictor = jobs + salary. Same skeleton!

---

## 🏗️ The Universal Recipe (memorize this skeleton)

```
Step 1. FIND DATA          → a CSV (or scrape one)
Step 2. CLEAN IT           → src/prep_data.py
Step 3. TRAIN & COMPARE    → src/train.py
Step 4. SAVE MODEL         → joblib.dump
Step 5. SERVE IT           → app/main.py
Step 6. SHOW IT            → README.md (+ deploy)
```

Your next project: copy DarPrice, change the data, rename things. Done.

---

## 💡 Project ideas (easy → harder)

| Project | Target (y) | Features (X) | Difficulty |
|---|---|---|---|
| **CarPrice MA** | car price in MAD | year, km, brand, fuel, city | easy (you already know how!) |
| RentalPrice | monthly rent | city, m², rooms, floor | easy |
| Salary predictor | salary | years exp, title, city | medium |
| Student score | exam grade | study hours, attendance | easy |
| House price CLASSIFIER | cheap / medium / expensive | same as DarPrice | medium (new skill: classification) |

Start with **CarPrice** — it's literally DarPrice with different column names.

---

## 📋 Step-by-step: your first solo build

### Step 1 — Data
Kaggle.com → search "used cars morocco" or "moroccan real estate" → download CSV.
Rule: needs ≥1000 rows and one numeric target column.

### Step 2 — Copy the skeleton
```bash
mkdir CarPrice && cd CarPrice
mkdir -p src app models data/raw data/processed
# copy prep_data.py, train.py, main.py from DarPrice and adapt names
```

### Step 3 — Adapt `prep_data.py`
- Change file paths
- Update `KEEP` list to YOUR columns
- Keep ALL traps lessons: drop leakage columns, filter impossible values, lowercase text

### Step 4 — Adapt `train.py`
- Change target column name (`y = df["price"]` → `y = df["your_target"]`)
- Run. Compare baseline vs LinearRegression vs RandomForest.
- ⚠️ If RandomForest doesn't clearly beat LinearRegression → your features are weak, go back to step 3, not step 4.

### Step 5 — Adapt `main.py`
- Rewrite the `House` class with YOUR fields → call it e.g. `Car`
- Nothing else changes. Seriously.

### Step 6 — README + share
Copy DarPrice's README structure: results table, how to run, API example.
Push to GitHub. Post on LinkedIn: what you built + metrics + screenshot of /docs.

---

## 🧭 When you're stuck (the universal debugging order)

1. READ the error message bottom line FIRST — it names the problem and line number
2. Print shapes: `print(df.shape)` after every pandas operation
3. Is it the DATA? (run prep, look at head) or the CODE? (read traceback)
4. Google the exact error text — someone had it before, always
5. Still stuck? Reduce the problem until it works, then grow it back

---

## 📈 After your 2nd project, level up with (in this order)

1. **Git & GitHub properly**: commit small, often, with clear messages
2. **GridSearchCV**: automatic hyperparameter search (replaces exercise-6-style manual tests)
3. **XGBoost**: usually beats RandomForest on tables — one-line swap!
   ```python
   from xgboost import XGBRegressor   # pip install xgboost
   ```
4. **Streamlit**: prettier than /docs for demos — an HTML form page in ~20 lines
5. **Docker**: package everything so it runs identically anywhere

Don't learn these NOW. Learn them when a project NEEDS them. Need-driven learning sticks; curiosity-driven lists don't.

---

## 🎓 Final truth

You didn't just learn Python libraries. You learned the SHAPE of real work:

> Get messy reality → force it into clean numbers → let math find patterns → serve those patterns to humans.

Every data scientist on Earth does exactly this, at every company, every day.
You've done it once completely. The second time alone is where mastery starts.

Bon courage ! 🇲🇦🐢
