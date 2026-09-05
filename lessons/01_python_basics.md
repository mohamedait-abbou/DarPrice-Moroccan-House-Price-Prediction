# Lesson 1: Python Basics (with your project's real code)

No libraries yet. Just pure Python. This is the foundation — everything else stands on this.

---

## 1. Variables = boxes with labels

A variable is a **box** where you put something, with a **name** on it.

```python
price = 1500000          # box "price" contains a NUMBER
city = "casablanca"      # box "city" contains TEXT (called a "string")
is_sold = True           # box "is_sold" contains True/False
area = 120.5             # number with decimal ("float")
```

You can look inside:

```python
print(city)        # casablanca
print(price * 2)   # 3000000   (you can do math!)
```

You can replace what's inside:

```python
price = 2000000    # the old 1,500,000 is gone forever
```

### The 4 types you need

| Type | Example | Meaning |
|---|---|---|
| `int` | `rooms = 3` | whole number |
| `float` | `area = 120.5` | decimal number |
| `str` | `city = "rabat"` | text (always in quotes) |
| `bool` | `sold = False` | only True or False |

---

## 2. Lists = many things in ONE box

```python
KEEP = ["city", "neighborhood", "property_type", "condition"]
```

This is from YOUR file `src/prep_data.py`. It's a list of column names.

Get items out by position (**starts at 0!**):

```python
KEEP[0]     # "city"         <- first is 0, NOT 1!
KEEP[1]     # "neighborhood"
KEEP[-1]    # "condition"    <- -1 means last one
```

Add to a list:

```python
KEEP.append("age")     # now the list has 5 items
```

Count items:

```python
len(KEEP)    # 5
```

Check if something is inside:

```python
"city" in KEEP       # True
"swimming_pool" in KEEP   # False
```

---

## 3. Dictionaries = box with NAMED slots

A list uses positions (0,1,2...). A dict uses **names**:

```python
house = {
    "city": "tanger",
    "area": 90,
    "rooms": 3,
}
```

Read one slot by its name:

```python
house["city"]     # "tanger"
house["area"]     # 90
```

Add or change a slot:

```python
house["price"] = 800000
```

**Where did you see this already?** In `app/main.py`, when someone calls your API they send EXACTLY this dict: `{"city": ..., "area": ...}`.

---

## 4. if / else = making decisions

```python
price = 600000

if price > 1000000:
    print("expensive house")
elif price > 300000:
    print("normal house")
else:
    print("cheap house")
```

⚠️ The spaces before `print` are **NOT decoration**. They are how Python knows what belongs inside the `if`. This is called **indentation** (4 spaces).

Real example from your project (`prep_data.py`):

```python
if c not in out.columns:
    out[c] = np.nan
```
Translation: "if this column doesn't exist in the table, create it empty."

---

## 5. Loops = repeat without repeating yourself

```python
for col in ["price", "area", "rooms"]:
    print(col)
```
Prints:
```
price
area
rooms
```
Translation: "for each item in this list, call it `col`, and do the indented thing."

The loop version of your cleaning code:

```python
for col in ["price", "area", "floor", "age"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
```
= "take each column name one by one, and force it to become numbers."
Without a loop you would write that line 4 times. With 50 columns? 50 times!

While-loop (repeat UNTIL something):

```python
count = 3
while count > 0:
    print(count)
    count = count - 1
# prints 3, 2, 1
```

---

## 6. Functions = machines you build

A function is a machine: **input goes in → work happens → output comes out**.

```python
def contract(df):
    return df[["city", "price", "area"]]
```

Pieces:
- `def` = "I am building a machine, its name is..."
- `contract` = the name (call it whatever makes sense)
- `(df)` = the INPUT (df = dataframe, a pandas table)
- `return` = what the machine gives back

Use it:

```python
small_table = contract(big_table)
```

Machine with several inputs and default values (from your `train.py`):

```python
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name} MAE: {mae:,.0f}")
```
Call it: `evaluate("RandomForest", forest, X_test, y_test)`.

### f-strings: putting variables INSIDE text

```python
name = "RandomForest"
mae = 592859
print(f"{name} made errors of {mae} MAD")
# RandomForest made errors of 592859 MAD
```
The `f` before the quote turns on this magic.
In your code: `f"{mae:,.0f}"` means "show mae with commas and no decimals" → `592,859`.

---

## 7. Imports = borrowing other people's machines

```python
from pathlib import Path        # take ONLY the Path tool
import pandas as pd             # take the whole pandas library, nickname pd
```

- `from X import Y` → take one specific tool out of the toolbox
- `import X as Z` → take everything, give it a short nickname

Your project's tools:
| Library | What it does |
|---|---|
| `pandas` (pd) | tables (Excel for Python) |
| `numpy` (np) | fast math on lists of numbers |
| `sklearn` | all machine learning |
| `joblib` | save/load trained models |
| `fastapi` | make a web API |
| `pathlib` | handle file paths |

---

## ✅ QUIZ — answer before looking at solutions!

1. What does this print?
```python
x = 10
x = x + 5
print(x)
```

2. What does this print?
```python
cities = ["fes", "rabat", "oujda"]
print(cities[1])
```

3. Write a function `double(price)` that returns price × 2.

4. What does this print?
```python
for c in ["a", "b"]:
    print(c + "!")
```

5. True or False: `KEEP[0]` gives the FIRST item of the list.

---

<details>
<summary>👉 Solutions (click / open only AFTER trying)</summary>

1. `15` (10 plus 5 goes back into the same box)
2. `rabat` (position 1 = second item!)
3. 
```python
def double(price):
    return price * 2
```
4. 
```
a!
b!
```
5. True. Position counting starts at 0.

</details>

---

**Next: Lesson 2 — pandas, the Excel of Python 🐼**
