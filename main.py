import Turnover
import Pricing
import streamlit as st
PAGES = {
    "Pricing": Pricing,
    "Turnover": Turnover
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
