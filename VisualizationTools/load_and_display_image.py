import matplotlib.pyplot as plt
import streamlit as st

def load_and_display_image(path:str, title:str="",subtext:str="", pos:st=st):
    '''
    :param col: streamlit column element
    '''
    pos.header(title)
    try:
        img = plt.imread(path)
        pos.image(img, use_column_width=True)
    except Exception:
        pos.error('No image')
    pos.markdown(f"<h5 style='text-align: center; color: withe;' >{subtext}</h5>", unsafe_allow_html=True)
					
    #pos.subheader(subtext)