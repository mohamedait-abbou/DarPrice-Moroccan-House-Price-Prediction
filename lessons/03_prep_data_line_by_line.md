# Lesson 3: `src/prep_data.py` — Line by Line 🔬

Open the real file next to this lesson. We read it together, top to bottom.
**Goal:** after this lesson you can edit this file WITHOUT fear.

---

## The mission of this file

```
2 messy CSV files  →  1 clean CSV file
```

That's ALL it does. Input: raw data. Output: `data/processed/houses_combined.csv`.

---

## Part A — Setup

```python
from pathlib import Path
import numpy as np
import pandas as pd
```
Borrow tools. (Lesson 1 & 2.)

```python
ROOT = Path(__file__).parent.parent
```
- `__file__` = path of THIS file → `.../DarPrice/src/prep_data.py`
- `.parent` → `.../DarPrice/src`
- `.parent.parent` → `.../DarPrice` ✅

So `ROOT / "data/raw/..."` works no matter WHERE you run the command from. This kills a whole family of bugs.

```python
KEEP = ["city", "neighborhood", "property_type", "condition", "age",
        "price", "area", "rooms", "bathrooms", "floor", "amenities_count"]
```
The **contract**: the exact list of columns we want in the final table.
CAPITALS = constant, don't change while running.

---

## Part B — `contract(df)`: force one shape

```python
def contract(df):
    out = df[[c for c in KEEP if c in df.columns]].copy()
    for c in KEEP:
        if c not in out.columns:
            out[c] = np.nan
    return out[KEEP]
```

Line by line:

1. `df[[c for c in KEEP if c in df.columns]]`
   This is a **list comprehension**: build a list inside `[...]`.
   Read it as: "keep only the columns from KEEP that actually exist".
   `.copy()` = work on a photocopy, don't touch the original table (avoids pandas warnings).

2. ```python
   for c in KEEP:
       if c not in out.columns:
           out[c] = np.nan
   ```
   "For each wanted column: if it's missing, create it full of holes (NaN)."
   Example: source 1 has no `condition` column → we create it empty, so both tables match.

3. `return out[KEEP]` → order columns EXACTLY like the KEEP list.

**Why all this?** `pd.concat` needs identical column names to stack two tables. Contract = both tables wear the same uniform 👕.

---

## Part C — `load_source_1()`: tame the MESSY source

Source 1 (`houses_data_eng.csv`) has everything stuffed inside ONE text column:

```
"Appartement à vendre Casablanca - Maarif"
   └── type ──────────┘└city┘  └neighborhood┘
```

### The magic knife (regex)

```python
m = df1["address"].str.extract(
    r"^(?P<type>.+?) à vendre (?P<city>.+?) - (?P<quarter>.+)$"
)
```

Don't panic. Read it as a sentence:
- `^` start of text
- `(?P<type>.+?)` → grab some text, NAME it "type" (the `.+?` means "any characters, shortest possible")
- ` à vendre ` → literal text separator
- `(?P<city>.+?)` → grab text, name it "city"
- ` - ` → literal separator
- `(?P<quarter>.+?)$` → grab text until the end (`$`), call it quarter

Result: columns `type`, `city`, `quarter` cut cleanly out of the address. ✂️

Then:

```python
df1["property_type"] = m["type"].str.strip().str.lower()
```
`.str.strip()` → remove extra spaces. `.str.lower()` → "Casablanca" and "casablanca" must be THE SAME word for the computer. Cleaning text = lowercase + strip. Always.

```python
df1 = df1.dropna(subset=["city"])
```
If the regex couldn't cut an address (weird format) → city is NaN → delete that row. Garbage in, garbage out? No thanks.

```python
if "ascenseur" in df1.columns:
    df1["amenities_count"] = (df1["ascenseur"].astype(str).str.strip().str.lower() == "yes").astype(float)
```
Turn "yes"/"no" into 1.0/0.0. Why float? Because later the model only eats NUMBERS.
The `if` protects us if the column doesn't exist.

Finally rename to contract names:
```python
df1 = df1.rename(columns={"new_price": "price", "surface": "area",
                          "chambres": "rooms", "salles de bains": "bathrooms"})
```

---

## Part D — `load_source_2()`: defuse the traps 💣

Source 2 is cleaner BUT contains **traps**:

### Trap 1: rentals mixed with sales
```python
df2 = df2[df2["transaction"] == "vente"].copy()
```
Rent prices are 10x smaller than sale prices. Mixing them = teaching the model lies.

### Trap 2: LEAKAGE ⚠️ (the most important lesson here)
```python
df2 = df2.drop(columns=["price_per_sqm", ...])
```
`price_per_sqm = price / area`. It CONTAINS the answer inside it!
If the model sees it: it just multiplies back and "predicts" perfectly — on paper. Real life: useless. This is called **data leakage**, and removing it is what separates pros from beginners.

### Trap 3: noise
`reference` = listing ID number. Random digits have zero meaning for price → drop.

### Then normalize like source 1:
```python
df2["amenities_count"] = df2["amenities"].fillna("").astype(str).str.split(",").str.len()
df2 = df2.rename(columns={"type": "property_type", "surface": "area"})
```

---

## Part E — `clean()`: merge + kill bad rows

```python
df = pd.concat([contract(load_source_1()), contract(load_source_2())], ignore_index=True)
```
Uniform on → stack them. One big table.

```python
df = df[~df["property_type"].isin(["bureau", "nan", ""])]
```
Remove offices (different market) and garbage values. `isin([...])` = "is one of these".

```python
for col in ["price", "area", "rooms", "bathrooms", "floor", "age"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
```
Force numbers. `errors="coerce"` = if conversion impossible ("N/A", "?"), put NaN instead of crashing.

```python
df = df.drop_duplicates()
df = df.dropna(subset=["price", "area"])
```
No double listings. No house without price/area — those can't be used at all.

```python
df = df[(df["price"] > 50_000) & (df["area"] >= 10)]
ppm2 = df["price"] / df["area"]
df = df[(ppm2 > 2_000) & (ppm2 < 80_000)]
```
Reality filter 🇲🇦:
- No houses at 500 MAD (scraping errors)
- Price per m² between 2,000 and 80,000 MAD → removes typos like area=5 or price=50M for 20m²
- Underscores in numbers (`50_000`) = same as 50000, just easier to READ.

```python
for col in ["city", "neighborhood", "property_type", "condition"]:
    df[col] = df[col].astype(str).str.strip().str.lower()
```
Final text cleaning pass.

---

## Part F — `main()` and the bottom lines

```python
def main():
    ...
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
```
- `mkdir(parents=True, exist_ok=True)` → create folder; no error if it exists already.
- Save the clean table.

```python
if __name__ == "__main__":
    main()
```
Translation: "If someone runs me directly (`python src/prep_data.py`) → execute main().
But if another file IMPORTS me (like train.py does) → DON'T run anything, just lend me your functions."
This line is standard Python hygiene. Memorize it.

---

## ✏️ EDIT PRACTICE (do these for real!)

1. Open `prep_data.py`, change `KEEP` by removing `"floor"` → run `python src/prep_data.py` → look at `houses_combined.csv` header. What changed?
2. Change the price filter to `100_000` → how many rows survive now (check `Clean shape:`)?
3. Add `"terrasse"` nowhere — instead print `df.head()` right after concat. Do it with `print(df.head())`.

Breaking things ON PURPOSE then fixing them is the fastest way to learn. You cannot break anything permanent — git will protect you later.

---

**Next: Lesson 4 — sklearn: how the model really learns 🧠**
