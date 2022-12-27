import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from matplotlib.pyplot import figure
figure(figsize=(10, 10), dpi=80)
from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg')

from statsmodels.tsa.arima.model import ARIMA
import pmdarima
import arch 
import base64
from simulate_garch import simulate_GARCH
import numpy as np
import ruptures as rpt

#st.set_page_config(layout="centered")



apptitle = 'CFA ARADEI CAPITAL'

col1, col2= st.columns(2)

with col1:st.image('ensa.png',width=500)

with col2:st.image('Aradei.jpg',width=400)


st.markdown('__________________________________________________________')
st.markdown('### Work realized by: Anwar Adnane,Hoda jaouhari, Ahmed Ouaboune, Yassin rzhif, Mouad Rhafir')
st.markdown("### Under the supervision of: Amina Tourabi")
st.markdown('__________________________________________________________')

#st.set_page_config(page_title=apptitle, page_icon=":chart_with_upwards_trend:")

# Title the app
st.title(' Management control project')

# @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data

# @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data




st.markdown('### Subject Company :  ARADEI CAPITAL  ')

#st.markdown('<center><img src="ensa.png" width="300"  height="100" alt="Ensa logo"></center>', unsafe_allow_html=True)

st.markdown('##')
st.markdown('__________________________________________________________')

st.markdown('## Analysis of the CAF')

data = pd.read_excel('caf.xlsx')
data= data.set_index('Année')

left_column, right_column = st.columns(2)
with left_column:
    st.text('Descriptive analysis')
    st.write(data)

with right_column :
    st.text('Descriptive analysis ')
    st.write(data.describe())


left_column, right_column = st.columns(2)
with left_column:
    st.text('Graphical analysis ')
    fig = plt.figure(figsize=(10, 5))
    plt.plot(data)
    st.pyplot(fig)

with right_column :
    st.text('Graphical analysis')
    fig = plt.figure(figsize=(10, 5))
    plt.boxplot(data) 
    st.pyplot(fig)
 
Predictions = st.button("Start the prediction program ")

if Predictions:
    data.loc['2022'] =None
    data.loc['2023'] =None

    def split(data ,size):
        train_data = data[:int(np.ceil(len(data)*size))]
        test_data = data[len(train_data):]
        return train_data, test_data
    # --------------------------------------------------------
    train_data1, test_data1 = split(data, 0.55)
    # --------------------------------------------------------

    #test_data1 =2*[1]

    arima_model_fitted1 = pmdarima.auto_arima(train_data1)
    arima_residuals1 = arima_model_fitted1.arima_res_.resid

    # fit a GARCH(1,1) model on the residuals of the ARIMA model
    garch1 = arch.arch_model(arima_residuals1, p=1, q=1);
    garch_fitted1 = garch1.fit();

    # Use ARIMA to predict mu
    Model = ARIMA(train_data1, order=arima_model_fitted1.order)
    Fited_model1 =  Model.fit()
    arima_predict1 = pd.DataFrame(columns=[])
    arima_predict1['predicted'] = Fited_model1.forecast(steps=len(test_data1))  #Fited_model.predict(start =len(sp_return_trn) ,end=len(sp_return_trn)+len(sp_return_tst)-1, dynamic=False)  
    arima_predict1['Date'] = test_data1.index # np.arange(0,60)
    arima_predict1= arima_predict1.set_index('Date')

    # Use GARCH to predict the residual
    sim_resid1, sim_variance1= simulate_GARCH(observations =len(test_data1) ,omega = garch_fitted1.params[1], alpha = garch_fitted1.params[2], beta = garch_fitted1.params[3]) 


    df1 = pd.DataFrame()
    df1['Real'] = data
    df1['ARIMA Prediction'] = pd.concat([pd.Series(garch_fitted1.resid+Fited_model1.predict()), arima_predict1.predicted + sim_resid1 ],ignore_index = True).values

#plot________________________________________________________
    fig = plt.figure(figsize=(10, 5))

    plt.plot(df1['ARIMA Prediction'])
    plt.plot(df1['Real'])
    plt.legend(['ARIMA Prediction','Real'])

    plt.axvspan(2013, 2019, alpha=0.2, color='darkseagreen')
    plt.axvspan(2019, 2023, alpha=0.2, color='cyan')
    st.text('This graph presents the CAF prediction for a 2 year horizon: ')
    st.pyplot(fig)

    
def to_excel_M(df1,df2,df3,df4,df5,df6,df7,df8):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df1.to_excel(writer, index=False, sheet_name='Single Tenants Assets')
    df2.to_excel(writer, index=False, sheet_name='Shopping centers')
    df3.to_excel(writer, index=False, sheet_name='Commercial Galleries')
    df4.to_excel(writer, index=False, sheet_name='Industrial Unit')
    df5.to_excel(writer, index=False, sheet_name='Assets total revenue')
    df6.to_excel(writer, index=False, sheet_name='Total Revenues')
    df7.to_excel(writer, index=False, sheet_name='Net Income')
    df8.to_excel(writer, index=False, sheet_name='FFO')


    workbook = writer.book
    worksheet1 = writer.sheets['Single Tenants Assets']
    worksheet2 = writer.sheets['Shopping centers']
    worksheet3 = writer.sheets['Commercial Galleries']
    worksheet4 = writer.sheets['Industrial Unit']
    worksheet5 = writer.sheets['Assets total revenue']
    worksheet6 = writer.sheets['Total Revenues']
    worksheet7 = writer.sheets['Net Income']
    worksheet8 = writer.sheets['FFO']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet1.set_column('A:A', None, format1)
    worksheet2.set_column('A:A', None, format1)
    worksheet3.set_column('A:A', None, format1)
    worksheet4.set_column('A:A', None, format1)
    worksheet5.set_column('A:A', None, format1)
    worksheet6.set_column('A:A', None, format1)
    worksheet7.set_column('A:A', None, format1)
    worksheet8.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data  

df = pd.read_excel ('data.xlsx')
ST = pd.read_excel('data.xlsx',sheet_name ='ST').dropna()
SHC = pd.read_excel('data.xlsx',sheet_name ='SHC').dropna()
CG = pd.read_excel('data.xlsx',sheet_name ='CG').dropna()
IU = pd.read_excel('data.xlsx',sheet_name ='IU').dropna()
TR1 = pd.read_excel('data.xlsx',sheet_name ='TR1').dropna()
TR2 = pd.read_excel('data.xlsx',sheet_name ='TR2').dropna()
NET_INC = pd.read_excel('data.xlsx',sheet_name ='NET_Income').dropna()
FFO = pd.read_excel('data.xlsx',sheet_name ='FFO').dropna()

hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
#st.table(ST.style.format({"E": "{:.2f}"}))
#  Single Tenants Assets : (visualisation) 
st.markdown('## Single Tenants Assets')
st.table(ST.applymap(lambda x: int(round(x, 0)) if isinstance(x, (int, float)) else x))
test=pd.read_excel('visualisation.xlsx',sheet_name ='ST')
ST_v =test.set_index('Single Tenants Assets')
fig = plt.figure(figsize=(10,6), tight_layout=True)
#fig, ax = plt.subplot()
#plotting
plt.plot(ST_v, 'o-', linewidth=2)
#customization
#plt.xticks(['2019', '2020','2021','2021','2021E','2021E','2021E','2022E'])
plt.xlabel('Years')
plt.ylabel('Revenues')
plt.title('Revenues and forecasting revenues troughtout the years')
plt.legend(title='Single Tenants Assets', title_fontsize = 13, labels=ST_v.columns,bbox_to_anchor=(1.,0.3), loc="lower left")
st.write(fig)



st.markdown('## Shopping centers')
st.table(SHC.applymap(lambda x: int(round(x, 0)) if isinstance(x, (int, float)) else x))
test = pd.read_excel('visualisation.xlsx',sheet_name ='SHC').dropna()
SHC_v=test.set_index('Shopping centers')
#plt.figure(figsize=(10,6), tight_layout=True)
fig, ax= plt.subplots(figsize=(10,6),tight_layout=True)
#plotting
ax.plot(SHC_v, 'o-', linewidth=2)
plt.xlabel('Years')
plt.ylabel('Revenues')
plt.title('Revenues and forecasting revenues troughtout the years')
plt.legend(title='Shopping centers', title_fontsize = 13, labels=SHC_v.columns,bbox_to_anchor=(1.,0.3), loc="lower left")
st.write(fig)


st.markdown('## Commercial Galleries')
st.table(CG.applymap(lambda x: int(round(x, 0)) if isinstance(x, (int, float)) else x).astype(str))
test = pd.read_excel('visualisation.xlsx',sheet_name ='CG').dropna()
CG_V=test.set_index('Commercial Galleries')
#plt.figure(figsize=(10,6), tight_layout=True)
fig, ax= plt.subplots(figsize=(10,6),tight_layout=True)
#plotting
ax.plot(CG_V, 'o-', linewidth=2)
plt.xlabel('Years')
plt.ylabel('Revenues')
plt.title('Revenues and forecasting revenues troughtout the years')
plt.legend(title='Commercial Galleries', title_fontsize = 13, labels=CG_V.columns,bbox_to_anchor=(1.,0.3), loc="lower left")
st.write(fig)

st.markdown('## Industrial Unit')
st.table(IU.applymap(lambda x: int(round(x, 0)) if isinstance(x, (int, float)) else x))
test = pd.read_excel('visualisation.xlsx',sheet_name ='IU').dropna()
IU_V=test.set_index('Industrial Unit')
#plt.figure(figsize=(10,6), tight_layout=True)
fig, ax= plt.subplots(figsize=(10,6),tight_layout=True)
#plotting
ax.plot(IU_V, 'o-', linewidth=2)
plt.xlabel('Years')
plt.ylabel('Revenues')
plt.title('Revenues and forecasting revenues troughtout the years')
plt.legend(title='Industrial Unit', title_fontsize = 13, labels=IU_V.columns,bbox_to_anchor=(1.,0.3), loc="lower left")
st.write(fig)

st.markdown('## Assets total revenue')
st.table(TR1.applymap(lambda x: int(round(x, 0)) if isinstance(x, (int, float)) else x).astype(str))
#st.markdown('## Shopping centers')
st.table(TR2.applymap(lambda x: int(round(x, 0)) if isinstance(x, (int, float)) else x))
test = pd.read_excel('visualisation.xlsx',sheet_name ='TR2').dropna()
IU_V=test.set_index('(KMAD)')
#plt.figure(figsize=(10,6), tight_layout=True)
fig, ax= plt.subplots(figsize=(10,6),tight_layout=True)
#plotting
ax.plot(IU_V, 'o-', linewidth=2)
plt.xlabel('Years')
plt.ylabel('Revenues')
plt.title('Revenues and forecasting revenues troughtout the years')
plt.legend(title='Industrial Unit', title_fontsize = 13, labels=IU_V.columns,bbox_to_anchor=(1.,0.3), loc="lower left")
st.write(fig)

st.markdown('## Revenues ')
st.table(NET_INC.applymap(lambda x: int(round(x, 0)) if isinstance(x, (int, float)) else x))
test = pd.read_excel('visualisation.xlsx',sheet_name ='Net_Income').dropna()
Net_Income =test.set_index('Years')
#plt.figure(figsize=(10,6), tight_layout=True)
fig, ax= plt.subplots(figsize=(15,9),tight_layout=True)
#plotting
ax.plot(Net_Income, 'o-', linewidth=2)
plt.xlabel('Years')
plt.ylabel('Revenues')
plt.title('Revenues and forecasting revenues troughtout the years')
plt.legend(title='Industrial Unit', title_fontsize = 13, labels=Net_Income.columns,bbox_to_anchor=(1.,0.3), loc="lower left")
fig1, ax1= plt.subplots(figsize=(15,9),tight_layout=True)
plt.legend(title_fontsize = 13, labels=Net_Income.columns,bbox_to_anchor=(1.,0.1), loc="lower left")
Net_Income.plot(ax=ax1,kind='bar', stacked=True,figsize=(15, 8),color=["#ADD8E6","#00BFFF"])
st.write(fig1)
st.write(fig)

st.markdown('## Fund From Operation')
st.table(FFO.applymap(lambda x: int(round(x, 0)) if isinstance(x, (int, float)) else x))
test = pd.read_excel('visualisation.xlsx',sheet_name ='FFO').dropna()
FFO1 =test.set_index('Years')
#plt.figure(figsize=(10,6), tight_layout=True)
fig, ax= plt.subplots(figsize=(15,9),tight_layout=True)
#plotting
#ax.plot(FFO, 'o-', linewidth=2,color='red')
plt.xlabel('Years')
plt.ylabel('Revenues')
plt.title('Revenues and forecasting revenues troughtout the years')
plt.legend(title='Industrial Unit', title_fontsize = 13, labels=Net_Income.columns,bbox_to_anchor=(1.,0.3), loc="lower left")
plt.legend(title_fontsize = 13, labels=Net_Income.columns,bbox_to_anchor=(1.,0.1), loc="lower left")
FFO1.plot.bar(ax=ax, stacked=True,figsize=(15, 8),color=["#00BFFF"])
st.write(fig)
df_xlsx = to_excel_M(ST,SHC,CG,IU,TR1,TR2,NET_INC,FFO)
st.download_button(label='📥 Download Current Result',
                            data=df_xlsx ,
                            file_name= 'FFO_prediction.xlsx')




    
    
