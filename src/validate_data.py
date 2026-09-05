import pandas as pd, sys
required={"state","city","locality","property_type","bhk","area_sqft","bathrooms","floor","total_floors","age_years","furnished","parking","balcony","price_inr"}
p=sys.argv[1] if len(sys.argv)>1 else "data/sample_housing_data.csv"
df=pd.read_csv(p)
missing=required-set(df.columns)
if missing: raise SystemExit(f"Missing columns: {sorted(missing)}")
if (df.price_inr<=0).any(): raise SystemExit("price_inr must be positive")
print(f"OK: {len(df):,} rows, {df.state.nunique()} states/UTs, {df.city.nunique()} cities.")
