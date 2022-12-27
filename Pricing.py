import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import base64

def app():


    apptitle = 'CFA ARADEI CAPITAL'

    st.image('ensa.png',width=500)
    #st.set_page_config(page_title=apptitle, page_icon=":chart_with_upwards_trend:")

    # Title the app
    st.title(' Management control project')

    # @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data

    # @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data
        
    


    st.markdown('### Subject Company :  ARADEI CAPITAL  ')

    #st.markdown('<center><img src="ensa.png" width="300"  height="100" alt="Ensa logo"></center>', unsafe_allow_html=True)
   
    st.markdown('##')
    st.markdown('__________________________________________________________')
        
    data = pd.read_excel('caf.xlsx')
    data= data.set_index('Année')
    st.write(data)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(data)
    st.pyplot(fig)

   
