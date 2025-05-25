import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import datetime
import pickle
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

#set up page configuration for streamlit
def main():
    st.set_page_config(
        page_title='Singapore Flat Resale Price Predictor',
        page_icon='🏠',
        initial_sidebar_state='expanded',
        layout='wide',
        menu_items={"about": 'This Streamlit application developed for Singapore flat Resale price prediction'}
    )

    # Display the page title at the top of your app
    st.title(':blue[Singapore Flat Resale Price Predictor]')

    #set up the sidebar with optionmenu
    selected = option_menu("🏢Singapore Flat Resale | Analysis and Predictive Modeling",
                                options=["Home","Get Prediction","Explore"],
                                icons=["house","lightbulb","bar-chart-line"],
                                orientation="horizontal")
        # Set up the information for 'Home' menu
    if selected == 'Home':
        title_text = '''<h1 style='font-size: 30px;text-align: center; color:grey;'>Singapore Resale Flat Prices Predicting</h1>'''
        st.markdown(title_text, unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1.5], gap="large")
        with col1:
            st.markdown("### :red[Skills Takeaway]:")
            st.markdown('<h5> Data Wrangling, EDA, Model Building, Model Deployment </h5>', unsafe_allow_html=True)

            st.markdown("### :red[Domain]:")
            st.markdown('<h5> Real Estate </h5>', unsafe_allow_html=True)

            st.markdown("### :red[Overview]:")
            st.markdown('''<h4>
                                <li> Collected and Processed Singapore HDB resale flat transaction data (1990 to present) using Python,<br>
                                <li>Cleaned and structured data for machine learning,<br>
                                <li>Analyzed pricing trends and predictions,<br>
                                <li>Developed a user-friendly web application for resale price predictions.
                        </h4>''', unsafe_allow_html=True)

            

        with col2:
            st.image("D:\project\singapore flat images.jpg", use_column_width=True)
            st.write("----")
            st.markdown("  ")



    #user input values for selectbox and encoded for respective features
    class option:

        option_months = ["January","February","March","April","May","June","July","August","September","October","November","December"]

        current_year = datetime.datetime.now().year
        option_year = [str(year) for year in range(1990, current_year + 1)]

        encoded_month= {"January" : 1,"February" : 2,"March" : 3,"April" : 4,"May" : 5,"June" : 6,"July" : 7,"August" : 8,"September" : 9,
                "October" : 10 ,"November" : 11,"December" : 12}

        option_town=['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH','BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
            'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST','KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG','SERANGOON',
            'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN','LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS','PUNGGOL']
        
        encoded_town={'ANG MO KIO' : 0 ,'BEDOK' : 1,'BISHAN' : 2,'BUKIT BATOK' : 3,'BUKIT MERAH' : 4,'BUKIT PANJANG' : 5,'BUKIT TIMAH' : 6,
            'CENTRAL AREA' : 7,'CHOA CHU KANG' : 8,'CLEMENTI' : 9,'GEYLANG' : 10,'HOUGANG' : 11,'JURONG EAST' : 12,'JURONG WEST' : 13,
            'KALLANG/WHAMPOA' : 14,'LIM CHU KANG' : 15,'MARINE PARADE' : 16,'PASIR RIS' : 17,'PUNGGOL' : 18,'QUEENSTOWN' : 19,
            'SEMBAWANG' : 20,'SENGKANG' : 21,'SERANGOON' : 22,'TAMPINES' : 23,'TOA PAYOH' : 24,'WOODLANDS' : 25,'YISHUN' : 26}
        
        option_flat_type=['1 ROOM', '2 ROOM','3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE','MULTI-GENERATION']

        encoded_flat_type={'1 ROOM': 0,'2 ROOM' : 1,'3 ROOM' : 2,'4 ROOM' : 3,'5 ROOM' : 4,'EXECUTIVE' : 5,'MULTI-GENERATION' : 6}

        option_flat_model=['2-ROOM','3GEN','ADJOINED FLAT', 'APARTMENT' ,'DBSS','IMPROVED' ,'IMPROVED-MAISONETTE', 'MAISONETTE',
                        'MODEL A', 'MODEL A-MAISONETTE','MODEL A2' ,'MULTI GENERATION' ,'NEW GENERATION', 'PREMIUM APARTMENT',
                        'PREMIUM APARTMENT LOFT', 'PREMIUM MAISONETTE','SIMPLIFIED', 'STANDARD','TERRACE','TYPE S1','TYPE S2']

        encoded_flat_model={'2-ROOM' : 0,'3GEN' : 1,'ADJOINED FLAT' : 2,'APARTMENT' : 3,'DBSS' : 4,'IMPROVED' : 5,'IMPROVED-MAISONETTE' : 6,
                    'MAISONETTE' : 7,'MODEL A' : 8,'MODEL A-MAISONETTE' : 9,'MODEL A2': 10,'MULTI GENERATION' : 11,'NEW GENERATION' : 12,
                    'PREMIUM APARTMENT' : 13,'PREMIUM APARTMENT LOFT' : 14,'PREMIUM MAISONETTE' : 15,'SIMPLIFIED' : 16,'STANDARD' : 17,
                    'TERRACE' : 18,'TYPE S1' : 19,'TYPE S2' : 20}
        
    #set up information for the 'get prediction' menu
    if selected == "Get Prediction":
        st.write('')
        title_text = '''<h2 style='font-size: 32px;text-align: center;color:grey;'>Resale Flat Price Prediction</h2>'''
        st.markdown(title_text, unsafe_allow_html=True)
        st.markdown("<h5 style=color:orange>To Predict the Resale Price of a Flat, Please Provide the Following Information:",unsafe_allow_html=True)
        st.write('')

        # creted form to get the user input 
        with st.form('prediction'):
            col1,col2=st.columns(2)
            with col1:

                user_month=st.selectbox(label='Month',options=option.option_months,index=None)

                user_town=st.selectbox(label='Town',options=option.option_town,index=None)

                user_flat_type=st.selectbox(label='Flat Type',options=option.option_flat_type,index=None)

                user_flat_model=st.selectbox(label='Flat Model',options=option.option_flat_model,index=None)

                floor_area_sqm=st.number_input(label='Floor area sqm (10 to 307")',min_value=10.0)

                price_per_sqm=st.number_input(label='Price Per sqm',min_value=100.00)

            with col2:
                year = st.selectbox(label='Year', options=option.option_year, index=None)
                
                block = st.number_input(label='Block (1 to 999)', min_value=1, max_value=999, step=1)
                
                lease_commence_date = st.text_input(label='Year of lease commence (1966 to 2020)', max_chars=4)
                
                remaining_lease = st.number_input(label='Remaining lease year (0 to 99)', min_value=0, max_value=99, step=1)
                
                years_holding = st.number_input(label='Years Holding (0 to 99)', min_value=0, max_value=99, step=1)

                c1,c2=st.columns(2)
                with c1:
                    storey_start=st.number_input(label='Storey start (1 to 50 ) ',min_value=1,max_value=50)
                with c2:
                    storey_end=st.number_input(label='Storey end (1 to 51)',min_value=1,max_value=51)
                
                st.markdown('<br>', unsafe_allow_html=True)

                button=st.form_submit_button('PREDICT PRICE',use_container_width=True)

                st.markdown("""
                            <style>
                            div.stButton > button:first-child {
                                background-color: #009999;
                                color: white;
                                width: 100%;
                            }
                            </style>
                        """, unsafe_allow_html=True)

        if button:
            with st.spinner("Predicting..."):

                #check whether user fill all required fields
                if not all([user_month,user_town,user_flat_type,user_flat_model,floor_area_sqm,price_per_sqm,year,block,
                            lease_commence_date,remaining_lease,years_holding,storey_start,storey_end]):
                    st.error("Please fill in all required fields.")

                else:
                    #create features from user input 
                    current_year=datetime.datetime.now().year

                    current_remaining_lease=remaining_lease-(current_year-(int(year)))
                    age_of_property=current_year-int(lease_commence_date)


                    month=option.encoded_month[user_month]
                    town=option.encoded_town[user_town]
                    flat_type=option.encoded_flat_type[user_flat_type]
                    flat_model=option.encoded_flat_model[user_flat_model]

                    floor_area_sqm_log=np.log(floor_area_sqm)
                    remaining_lease_log=np.log1p(remaining_lease)
                    price_per_sqm_log=np.log(price_per_sqm)

                    #opened pickle model and predict the resale price with user data
                    with open('Decisiontreemodel.pkl','rb') as files:
                        model=pickle.load(files)
                    
                    user_data=np.array([[month, town, flat_type, block, flat_model, lease_commence_date, year, storey_start,
                                        storey_end, years_holding, current_remaining_lease, age_of_property, floor_area_sqm_log, 
                                        remaining_lease_log,price_per_sqm_log ]])

                    predict = model.predict(user_data)
                    resale_price_usd = np.exp(predict[0])
                    resale_price_inr = resale_price_usd * 61.68

                    # Display the predicted selling price in USD and INR
                    st.subheader(f"Predicted Resale Price is: :green[$ {resale_price_usd:.2f}]")
                    st.subheader(f"Predicted Resale Price in INR: :green[₹ {resale_price_inr:.2f}]")

    # set up the information for 'Home' menu
    if selected == "Explore":
        st.markdown('<br>', unsafe_allow_html=True)  # Add some space before the topic
        st.subheader(':red[About Housing & Development Board]')
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info('''The **Housing & Development Board (HDB)** is Singapore’s statutory board under the Ministry of National Development, responsible for public housing.

- 🏗️ **Established** in **1960**, following the transition from the **Singapore Improvement Trust (SIT)**.
- 🛖 Initially focused on building **emergency housing** and **resettling kampong residents**.
- 🧱 In the **1990s and 2000s**, introduced:
  - **Upgrading programs** for mature estates  
  - **New housing types** for varying income groups  
  - Partnerships with **private developers**
- 🔄 In **2003**, reorganized to better address the **modern housing market**.

The HDB continues to play a vital role in shaping Singapore’s housing landscape through planning, development, and resale regulation.
            ''')

    st.markdown(" ")
    st.markdown(" ")

if __name__ == "__main__":
    main()
