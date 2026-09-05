# Lesson 4: scikit-learn — How the Model Really Learns 🧠

This lesson decodes `src/train.py`. Same method: file open, read together.

---

## 1. The vocabulary (memorize these 6 words)

| Word | Meaning | In DarPrice |
|---|---|---|
| **feature** | a clue: input info about a house | area, city, rooms... |
| **target** (`y`) | the answer we want to guess | price |
| **fit** | study phase | `model.fit(X_train, y_train)` |
| **predict** | guess on new data | `model.predict(X_test)` |
| **MAE** | average size of the error in MAD | 592,859 MAD |
| **R²** | % of the mystery explained (0→1) | 0.890 |

---

## 2. The split — why hide data from your own model?

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- 80% → student studies these (`train`)
- 20% → sealed envelope, opened only at exam (`test`)

**Why?** If you test on seen houses, a model that just MEMORIZED gets perfect scores but fails in real life. This fake skill is called **overfitting**.

- `random_state=42`: randomness with a fixed seed = same shuffle every run = reproducible results. Any number works; 42 is a programmer joke.

## 3. ColumnTransformer = the sorting table

```python
num_cols = X.select_dtypes(include=["number", "float", "int"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])
```
Numbers go left belt, text goes right belt. Each belt has its own cleaning pipeline.

## 4. The two belts

### Numbers belt:
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # fill holes with median value
    ("scaler", StandardScaler()),                    # shrink all to same scale
])
```

Why scale? Compare `area=200` vs `rooms=3`. To a computer, 200 looks 60x more important than 3 — just because it's bigger! StandardScaler puts everything on equal footing (average of values becomes 0).

### Text belt:
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),   # fill holes with most common word
    ("encoder", OneHotEncoder(handle_unknown="ignore")),    # words -> lightbulbs
])
```

OneHotEncoder example:

```
city = casablanca  →  city_casablanca=1, city_rabat=0, city_fes=0
city = rabat       →  city_casablanca=0, city_rabat=1, city_fes=0
```

One lightbulb ON, rest OFF. Why not city=1,2,3? Because the model would think fes < rabat < casablanca like sizes! Cities have no order.

`handle_unknown="ignore"`: prediction-time new city (say "agadir" never seen in training) → don't crash, all bulbs off.

## 5. Pipeline = glue it together

```python
forest = Pipeline([
    ("pre", pre),
    ("reg", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
])
```

Now cleaning + brain are ONE object. When you later do `model.predict(new_house)`, the SAME cleaning runs automatically. No manual steps = no mistakes.

**This is why the API can work with one line:** the .joblib file contains cleaner+brain frozen together.

## 6. RandomForest = 100 voting students 🗳️

LinearRegression: draws one straight recipe. Simple but rigid.
RandomForest: builds **100 decision trees**, each sees slightly different data, each gives an opinion → final answer = their AVERAGE.

A decision tree looks like:
```
area > 90 m²?
├─ YES → city == casablanca?
│        ├─ YES → predict ~1.5M MAD
│        └─ NO  → predict ~1.1M MAD
└─ NO  → predict ~700k MAD
```

- `n_estimators=100` → number of trees
- `n_jobs=-1` → use ALL CPU cores (faster)
- `random_state=42` → same forest every time

## 7. The exam and its grade

```python
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

**MAE** = for each test house: `|real - guessed|`, then average. Your result ~593k means: typical error is about half a million dirhams. Sounds big — but houses cost millions; relative error ≈ 10-15%.

**R²** compares you against a lazy guy who always guesses the average:
- R² = 1 → perfect
- R² = 0 → no better than the lazy guy
- R² < 0 → WORSE than lazy guy 😅

Your R²=0.89: model explains ~89% of price variation.

**Golden comparison table (always show baseline too):**
| Model | MAE | R² |
|---|---|---|
| Lazy mean-guesser | 2.1M | 0.00 |
| LinearRegression | 656k | 0.888 |
| RandomForest | **593k** | **0.890** |

## 8. Save & reload

```python
joblib.dump(forest, MODEL_PATH)     # freeze to disk 💾
# later, in app/main.py:
model = joblib.load("models/darprice_randomforest.joblib")   # thaw 🔓
```

---

## ✅ QUIZ

1. Why must `price_per_sqm` be deleted before training?
2. What does `test_size=0.2` mean?
3. Why OneHotEncoder instead of numbering cities 1,2,3?
4. Model has MAE 50k and R² = -0.3. Good or bad? Why?
5. What happens if a house from "oujda" arrives at predict time but oujda wasn't in training?

<details>
<summary>👉 Solutions</summary>

1. Leakage: the answer (price) hides inside it → fake perfect score, useless in real life.
2. 20% of rows become the hidden exam set; 80% for training.
3. Numbers imply order/size between cities that doesn't exist. Lightbulbs carry no order.
4. Bad! Negative R² = worse than always guessing the mean, despite small MAE (probably tiny prices dataset).
5. Nothing crashes: handle_unknown="ignore" → all city bulbs off, model guesses using other clues only.

</details>

---

**Next: Lesson 5 — FastAPI: your model goes online 🌍**
