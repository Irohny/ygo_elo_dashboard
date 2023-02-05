import streamlit as st

def insert_layout_htmls():
    st.set_page_config(page_title='YGO-Elo-Dashboard', page_icon=':trophy:' ,layout='wide')
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .row_heading.level0 {display: none;}
            .blank {display: none;}
            .big-font {fontsize:200px ;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)