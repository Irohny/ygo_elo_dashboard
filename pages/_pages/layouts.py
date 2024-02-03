import streamlit as st

def insert_layout_htmls():
    st.set_page_config(page_title='YGO-Elo-Dashboard', page_icon=':trophy:' ,layout='wide')
    st.sidebar.markdown("<sub style='text-align: center; color: withe;' >Version 4.2.0</sub>", unsafe_allow_html=True)
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
    

def insert_metric_style():
    st.markdown("""
        <style>
        div[data-testid="metric-container"] {
        background-color: rgba(0, 131, 184, .7);
        border: 1px solid rgba(0, 131, 184, .7);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        color: rgb(255, 255, 255);
        overflow-wrap: break-word;
        }

        /* breakline for metric text         */
        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        color: white;
        font-size: 175% !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown(
        '''
        <style>
        .streamlit-expanderHeader {
            background-color: rgba(0, 131, 184, .7);
            color: white; # Adjust this for expander header color
        }
        .streamlit-expanderContent {
            background-color: rgba(0, 131, 184, .7);
            color: white; # Expander content color
        }
        </style>
        ''', unsafe_allow_html=True)

def insert_container_style():
    css_body_container = f'''
        <style>
        [data-testid="stSidebar"] + section [data-testid="stVerticalBlock"] div:nth-of-type(1)
        [data-testid="stVerticalBlock"] {{
                padding: 5% 5% 5% 10%;
                background-color:rgba(28, 131, 225, 0.2);
                }}
        </style>
        '''
    st.markdown(css_body_container, unsafe_allow_html=True)