# Data

## What is committed

`data/processed/houses_combined.csv` **is committed** on purpose so a fresh clone
can train and serve immediately without downloading anything.

The trained model (`models/darprice_model.joblib`), its metadata
(`models/metadata.json`), `metrics.json`, and the `reports/` folder are committed
for the same reason: the whole project is reproducible from a clone.

## Raw sources

The raw scraped files are **not** committed (they are large and may have license
restrictions). To re-run `src/prep_data.py`, place these two files in `data/raw/`:

| File | Rows | Notes |
|---|---|---|
| `houses_data_eng.csv` | 4,675 | Listings from a Moroccan property portal. `address` carries type/city/neighborhood, and `Type`/`City`/`Nighberd` are used when present. |
| `morocco_houses_dataset.csv` | 10,000 | Tabular sales + rental listings with `transaction`, `price`, `surface`, amenities, etc. Only `transaction == "vente"` is kept. |

Expected columns are handled defensively in `src/prep_data.py`, but the two
sources must keep their original names.

### Data quality notes

- **Source 1 is highly duplicated.** Its 4,675 rows collapse to only **24 unique
  feature combinations** (the same listings repeat ~187 times). `clean()` removes
  exact duplicates, so those copies do not leak between train and test.
- **Source 2 is cleaner** and contributes the vast majority of the final rows.
- **Leakage:** `price_per_sqm` is a direct function of the target and is dropped.
  `reference`, `description`, and `listing_date` are identifiers/free text and are
  also dropped.
- **Outliers:** price ≤ 50,000 MAD, area < 10 m², and price/m² outside
  2,000–80,000 MAD are removed. Age must be 0–200, floor −10–200, and rooms /
  bathrooms must be non-negative.
- **Text normalization:** accents and case are stripped so `fès`/`fes` and
  `meknès`/`meknes` become one category.

## Final schema

`data/processed/houses_combined.csv` — **7,014 rows × 11 columns**:

| Column | Type | Meaning |
|---|---|---|
| `city` | text | Normalized city name |
| `neighborhood` | text | Normalized neighborhood/quarter |
| `property_type` | text | appartement, villa, studio, maison, duplex, riad |
| `condition` | text | neuf, bon etat, tres bon etat, excellent etat, a renover |
| `age` | number | Building age in years (NaN when unknown) |
| `price` | number | Sale price in MAD (**target**) |
| `area` | number | Surface in m² |
| `rooms` | number | Number of bedrooms |
| `bathrooms` | number | Number of bathrooms |
| `floor` | number | Floor number |
| `amenities_count` | number | Count of listed amenities |

## Rebuild the dataset

```bash
python -m src.prep_data
```

Output: `data/processed/houses_combined.csv` (7,014 rows × 11 columns) and a
printed cleaning funnel showing how many rows each rule removed.

## Rebuild the model

```bash
python -m src.train
```

Outputs:

- `models/darprice_model.joblib` — full sklearn `Pipeline` (feature engineering + preprocessing + regressor)
- `models/metadata.json` — valid categories/ranges for the API and UI
- `metrics.json` — CV scores, held-out metrics, best params, feature importance, versions
- `reports/` — error analysis CSV/JSON and charts

## Licensing

The raw datasets were collected from public Moroccan property listings for
educational use. Before redistributing them, check the original source's terms.
The code in this repository is MIT-licensed; the data is not covered by that
license.
