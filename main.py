import langchain_helper as lch
import streamlit as st

st.title("Pet Name Generator")

animal_type = st.sidebar.text_input("What is your pet type?")
pet_color = st.sidebar.text_input("What is your pet color?")

if st.sidebar.button("Generate Name"):
    response = lch.generate_pet_name(animal_type, pet_color)
    st.write(response)
