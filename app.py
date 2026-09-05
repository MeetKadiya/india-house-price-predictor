import json, sys
from pathlib import Path
import joblib, pandas as pd, streamlit as st
import plotly.express as px

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.train import train

st.set_page_config(page_title="India House Price Predictor", page_icon="🏠", layout="wide")

@st.cache_data
def load_locations():
    return json.loads((ROOT/"locations.json").read_text(encoding="utf-8"))

@st.cache_data
def load_data():
    return pd.read_csv(ROOT/"data"/"sample_housing_data.csv")

def money(x):
    x=float(x)
    if x>=1e7: return f"₹{x/1e7:.2f} Cr"
    if x>=1e5: return f"₹{x/1e5:.2f} L"
    return f"₹{x:,.0f}"

locations=load_locations()
df=load_data()

st.title("🏠 India House Price Predictor")
st.caption("India-wide • State → City → Locality • ML-based property estimate")
st.warning("Demo mode: the included dataset is synthetic. Replace it with licensed/recent real property data before treating predictions as market estimates.")

if not (ROOT/"models"/"house_price_model.joblib").exists():
    with st.spinner("Training initial model..."):
        train()
    st.success("Initial model trained.")

model=joblib.load(ROOT/"models"/"house_price_model.joblib")
meta=joblib.load(ROOT/"models"/"metadata.joblib")
metrics=pd.read_csv(ROOT/"models"/"metrics.csv")

tab1,tab2,tab3=st.tabs(["🔮 Predict","📊 Market Analytics","🧪 Model"])

with tab1:
    c1,c2,c3=st.columns(3)
    with c1:
        state=st.selectbox("State / Union Territory", list(locations.keys()))
    with c2:
        city=st.selectbox("City", locations[state])
    with c3:
        localities=sorted(df.loc[(df.state==state)&(df.city==city),"locality"].unique().tolist())
        locality=st.selectbox("Locality", localities if localities else [f"{city} Central"])

    c1,c2,c3,c4=st.columns(4)
    with c1: prop=st.selectbox("Property type",["Apartment","Independent House","Villa"])
    with c2: bhk=st.number_input("BHK",1,10,3)
    with c3: area=st.number_input("Area (sq.ft)",250,10000,1800,50)
    with c4: baths=st.number_input("Bathrooms",1,10,2)
    c1,c2,c3,c4=st.columns(4)
    with c1: floor=st.number_input("Floor",0,100,5)
    with c2: total=st.number_input("Total floors",1,100,12)
    with c3: age=st.number_input("Property age (years)",0,100,5)
    with c4: furnished=st.selectbox("Furnishing",["Unfurnished","Semi-Furnished","Furnished"])
    c1,c2=st.columns(2)
    with c1: parking=st.number_input("Parking spaces",0,10,1)
    with c2: balcony=st.number_input("Balconies",0,10,2)

    if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
        row={"state":state,"city":city,"locality":locality,"property_type":prop,"bhk":bhk,
             "area_sqft":area,"bathrooms":baths,"floor":floor,"total_floors":total,
             "age_years":age,"furnished":furnished,"parking":parking,"balcony":balcony}
        price=float(model.predict(pd.DataFrame([row]))[0])
        err=float(meta["interval_80"])
        low=max(0,price-err); high=price+err
        a,b,c=st.columns(3)
        a.metric("Estimated price",money(price))
        b.metric("Estimated ₹/sq.ft",f"₹{price/area:,.0f}")
        c.metric("Model range",f"{money(low)} – {money(high)}")
        st.info(f"Best model: {meta['best_model']}. The range is an empirical validation-error allowance, not a formal confidence interval.")

with tab2:
    st.subheader("Dataset overview")
    a,b,c,d=st.columns(4)
    a.metric("Rows",f"{len(df):,}")
    b.metric("States/UTs",df.state.nunique())
    c.metric("Cities",df.city.nunique())
    d.metric("Median price",money(df.price_inr.median()))
    city_stats=df.groupby("city").agg(avg_price=("price_inr","mean"), median_price=("price_inr","median"), listings=("price_inr","size")).reset_index()
    fig=px.bar(city_stats.nlargest(20,"avg_price"),x="city",y="avg_price",title="Top 20 cities by average demo price")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(df.head(100),use_container_width=True)

with tab3:
    st.subheader("Model comparison")
    st.dataframe(metrics.style.format({"mae":"₹{:,.0f}","rmse":"₹{:,.0f}","r2":"{:.3f}"}),use_container_width=True)
    st.write(f"Selected model: **{meta['best_model']}**")
    if st.button("🔄 Retrain model"):
        with st.spinner("Retraining..."):
            train()
        st.cache_data.clear()
        st.success("Model retrained. Reload the page.")

st.sidebar.markdown("### Data workflow")
st.sidebar.write("1. Replace sample_housing_data.csv with licensed/recent data")
st.sidebar.write("2. Keep the required column names")
st.sidebar.write("3. Retrain")
st.sidebar.write("4. Validate by city/locality")
st.sidebar.markdown("**Never use synthetic demo predictions for financial decisions.**")
