import streamlit as st

def header_styling():
    styles = {
        "nav": {
            "justify-content": "left",
        },
        "img": {
            "padding-right": "20px",
            "height":"40"
        },
         "div": {
            "max-width": "50rem",
        },
        "span": {
            "border-radius": "0.5rem",
            "margin": "0 0.125rem",
            "padding": "0.4375rem 0.625rem",
        },
        "active": {
            "background-color": "rgba(255, 255, 255, 0.25)",
        },
        "hover": {
            "background-color": "rgba(255, 255, 255, 0.35)",
        },
    }
    return styles
def sidebar(version:str='0.0.0'):
    st.sidebar.markdown(f"<sub style='text-align: center; color: withe;' >Version {version}</sub>", unsafe_allow_html=True)
    
def insert_layout_htmls():
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)
    st.markdown( """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .row_heading.level0 {display: none;}
            .blank {display: none;}
            .big-font {fontsize:200px ;}
            </style>
            """, unsafe_allow_html=True)
    
    

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