import streamlit as st

def check_roles(roles_allowed):
    return any([r in st.session_state['user_roles'] for r in roles_allowed])
