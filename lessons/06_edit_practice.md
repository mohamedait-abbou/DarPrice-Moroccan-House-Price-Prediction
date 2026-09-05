# Lesson 6: Edit Practice — Touch the Real Code ✏️

Reading ≠ knowing. Now you CHANGE your project. Each exercise is small,
safe, and teaches one thing. Do them in order. Break → run → observe.

> Golden safety: nothing here is permanent. If it breaks, undo (Ctrl+Z) or ask me to fix it.

---

## Exercise 1 — Warm up in `prep_data.py` 🌡️
**Goal:** prove to yourself you can change output.

1. Open `src/prep_data.py`, find `KEEP`
2. Remove `"floor"` from the list
3. Run: `python src/prep_data.py`
4. Check: does `floor` still exist in `data/processed/houses_combined.csv`?

**What you learned:** KEEP is the master control of columns. One edit = whole pipeline changes.

---

## Exercise 2 — Change a filter 🔍
**Goal:** see how cleaning rules shape the dataset.

1. In `clean()`, change `(df["price"] > 50_000)` to `(df["price"] > 200_000)`
2. Run prep. Note the new "Clean shape" — compare with before (~7,014)
3. Put back `50_000`

**Think:** why did rows disappear? Who decides what counts as "real" data? YOU do. This is called data curation and it's 80% of ML work.

---

## Exercise 3 — Retrain after changes 🧠
**Goal:** full loop: data change → model change.

1. After exercise 2's filter is restored, retrain: `python src/train.py`
2. Compare MAE/R² with the README table. Same? Tiny differences are normal (duplicates/outliers order).

**What you learned:** train.py always reads the CURRENT processed CSV. Pipeline = chain reaction.

---

## Exercise 4 — Add an endpoint ⚡
**Goal:** first real feature you ADD (not just modify).

In `app/main.py` add:

```python
@app.get("/health")
def health():
    return {"status": "ok", "model": "randomforest"}
```

Run `uvicorn app.main:app --reload` → visit http://localhost:8000/health

**What you learned:** adding a route = decorator + function + return dict. That easy.

---

## Exercise 5 — Make predict smarter 💡
**Goal:** enrich the response.

Change `/predict` return to:

```python
return {
    "predicted_price_MAD": round(price),
    "price_per_m2_MAD": round(price / house.area),
}
```

Test in /docs with area=100.

**What you learned:** responses are just dicts — add anything useful. And pydantic gives you typed access (`house.area`) for free.

---

## Exercise 6 — Playground experiment (notebook) 📓
**Goal:** understand hyperparameters by FEELING them.

In `test_code.ipynb` cell 7, try:
- `n_estimators=10` → train → note MAE
- `n_estimators=300` → train → note MAE

**Expected feeling:** more trees = slightly better but slower. Diminishing returns. This is hyperparameter tuning — normally done with GridSearchCV, but manual experiments build intuition first.

---

## Exercise 7 — The debugging workout 🐞
**Goal:** learn that errors are FRIENDS with addresses.

Predict on purpose each of these in /docs and READ the error message fully:

1. Send `"area": "big"` (text instead of number)
2. Delete `"city"` from the request

For each: which part rejected it? What line tells you exactly what's wrong?

**Skill unlocked:** reading error messages calmly = top skill separating juniors who freeze from juniors who fix.

---

## ✅ Checklist — you can now say:

- [ ] I changed input data and saw output change (ex. 1–3)
- [ ] I added a new API route alone (ex. 4)
- [ ] I enriched an existing route (ex. 5)
- [ ] I tuned a model parameter and measured impact (ex. 6)
- [ ] I read real error messages without panic (ex. 7)

Done all? You're ready for **Lesson 7: building YOUR next project alone** 🚀
