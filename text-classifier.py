import streamlit as st
from frame import setup as su


st.header('NB Text Classifier')
text = st.text_input('Enter your text:')
if st.button("Predict"):
    vec = su.vector.transform([text]).toarray()
    st.write('Label:',str(list(su.naivebayes.predict(vec))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))
