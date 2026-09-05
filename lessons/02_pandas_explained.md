# Lesson 2: pandas — The Excel of Python 🐼

pandas is THE library for tables. Your whole project is built on it.
Nickname: `pd`. Every command looks like `pd.something()`.

---

## 1. What a DataFrame is

A **DataFrame** = one table, like an Excel sheet:

```
        city    area   price
0   casablanca   100   1200000
1       rabat     80   1100000
2      tanger    150   1300000
```

- Columns have names (`city`, `area`...)
- Rows have numbers (0, 1, 2) — called the **index**
- In your code, this table is always called `df`

## 2. Read and save tables

```python
df = pd.read_csv("data/raw/morocco_houses_dataset.csv")
```
One line: CSV file → table in memory. This is how your project starts.

```python
df.to_csv("data/processed/houses_combined.csv", index=False)
```
Table → back to a file. `index=False` means "don't add an extra useless number column".

## 3. Look at your table (do this ALWAYS first)

```python
df.shape          # (rows, columns) e.g. (10000, 16)
df.head()         # show first 5 rows
df.columns        # names of all columns
df["price"].mean()    # average price
df.describe()     # stats about every numeric column
```

**Golden rule:** after ANY operation on `df`, print `df.shape` to see what happened. Did rows disappear? Which columns exist now?

## 4. Selecting: columns and rows

```python
y = df["price"]                  # ONE column -> like a list
X = df.drop(columns=["price"])   # EVERYTHING EXCEPT price
df[["city", "price"]]            # SEVERAL columns (note double [[ ]])
df[df["city"] == "rabat"]        # only ROWS where city is rabat
```

That last line reads as: "give me the table, WHERE city equals rabat".
This pattern `df[CONDITION]` is called **filtering**. You use it everywhere in `prep_data.py`:

```python
df2 = df2[df2["transaction"] == "vente"]     # keep only sales
df = df[(df["price"] > 50000)]               # keep only prices above 50k
df = df[~df["property_type"].isin(["bureau"])]   # ~ means NOT: remove bureaus
```

## 5. Renaming columns

```python
df = df.rename(columns={"surface": "area", "chambres": "rooms"})
```
Dict on purpose: `"old_name": "new_name"`.
Why? Source 1 calls it `surface`, source 2 too — but you want ONE name `area` everywhere so both tables can merge.

## 6. Creating new columns

```python
df["amenities_count"] = df["amenities"].fillna("").str.split(",").str.len()
```
Read it piece by piece:
1. `df["amenities"]` → the amenities text column, e.g. `"Ascenseur,Parking,Terrasse"`
2. `.fillna("")` → empty slots become `""` (empty text) instead of NaN
3. `.str.split(",")` → cut by comma → `["Ascenseur", "Parking", "Terrasse"]`
4. `.str.len()` → count pieces → `3`
5. `df["amenities_count"] = ...` → store result in NEW column

`.str` before any method means "apply this to TEXT". Numbers can't split.

Another real example from your notebook:
```python
ppm2 = df['price'] / df['area']       # price per m²
df = df[(ppm2 > 2000) & (ppm2 < 80000)]
```
`&` = AND. Both conditions must be true.

## 7. Missing values (NaN)

NaN = "Not a Number" = a hole in the table.

```python
df.isna().sum()              # count holes per column
df.dropna(subset=["price"])  # DELETE rows with no price
df["age"] = df["age"].fillna(15)   # fill holes with 15
```

Later, sklearn's `SimpleImputer` fills holes automatically during training — but deleting impossible rows (no price!) must happen BEFORE.

## 8. Combining two tables

```python
big = pd.concat([table1, table2], ignore_index=True)
```
= stack them vertically: table1 on top of table2.
`ignore_index=True` = renumber rows 0,1,2,... fresh.

This is exactly how DarPrice merges its two data sources.

⚠️ They can only stack if they have the SAME column names. That's why your `contract()` function exists — it forces both tables into the same shape first!

## 9. Grouping (the report machine)

```python
df.groupby("city")["price"].median()
```
= split rows by city, take each group's median price. Result:

```
casablanca    1350000
rabat         1500000
tanger          900000
```

Your notebook uses this to print "the #1 most expensive city" 📊.

---

## ✅ QUIZ

Given:
```python
df = pd.DataFrame({
    "city": ["fes", "fes", "rabat"],
    "price": [700000, 900000, 2000000],
    "area": [80, 100, 200],
})
```

1. How do you get ONLY the `price` column?
2. How do you get only the fes houses?
3. What does `df.shape` give?
4. Write one line that creates a new column `price_per_m2`.
5. What does `~` do in `df[~(df["city"] == "fes")]`?

<details>
<summary>👉 Solutions</summary>

1. `df["price"]`
2. `df[df["city"] == "fes"]`
3. `(3, 3)` → 3 rows, 3 columns
4. `df["price_per_m2"] = df["price"] / df["area"]`
5. It flips the condition → gives you all houses NOT in fes.

</details>

---

**Next: Lesson 3 — we open `src/prep_data.py` and decode it LINE BY LINE 🔬**
