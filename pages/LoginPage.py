import streamlit as st
import time

class LoginPage:
    def __init__(self,):
        col = st.columns([1,4,1])
        col[1].title(':trophy: Login :trophy:', anchor='anchor_tag')
        if not st.session_state['login']:
            form = col[1].form('Login')
            form.text_input('Name:', key='login_name')
            form.text_input('Passwort:', type='password', key='login_pwd')
            form.form_submit_button('Anmelden', on_click=self.__proof_login)
        else:
            col[1].success('Du bist eingeloggt')
            col[1].button('Abmelden', on_click=self.__logoff)

    def __proof_login(self,):
        if not st.session_state['login_name'] in st.secrets['users']:
            self.__login_feedback(False)
            return
        status = st.secrets[st.session_state['login_name']]['pwd'] == st.session_state['login_pwd']
        self.__login_feedback(status)
        st.session_state['login'] = status
        st.session_state['user_roles'] = st.secrets[st.session_state['login_name']]['role']
        st.rerun()

    def __login_feedback(self, status:bool):
        if status:
            st.success('Anmeldung efolgreich')
        else:
            st.error('Anmeldung fehlgeschlagen')
        time.sleep(2)
        
    def __logoff(self):
        st.session_state['login'] = False
        st.rerun()
