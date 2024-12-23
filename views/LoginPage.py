import streamlit as st
import time


def proof_login():
    if not st.session_state["login_name"] in st.secrets["users"]:
        login_feedback(False)
        return
    status = (
        st.secrets[st.session_state["login_name"]]["pwd"]
        == st.session_state["login_pwd"]
    )
    login_feedback(status)
    st.session_state["login"] = status
    st.session_state["user_roles"] = st.secrets[st.session_state["login_name"]]["role"]
    st.session_state["user_name"] = st.session_state["login_name"]
    st.rerun()


def login_feedback(status: bool):
    if status:
        st.success("Anmeldung efolgreich")
    else:
        st.error("Anmeldung fehlgeschlagen")
    time.sleep(2)


def logoff():
    st.session_state["login"] = False
    st.rerun()


##################################
# Page Layout
##################################
col = st.columns([1, 4, 1])
col[1].title(":trophy: Login :trophy:", anchor="anchor_tag")
if not st.session_state["login"]:
    form = col[1].form("Login")
    form.text_input("Name:", key="login_name")
    form.text_input("Passwort:", type="password", key="login_pwd")
    form.form_submit_button("Anmelden", on_click=proof_login)
else:
    col[1].success(f"Hallo {st.session_state['user_name']}")
    col[1].success("Du bist eingeloggt")
    col[1].button("Abmelden", on_click=logoff)
