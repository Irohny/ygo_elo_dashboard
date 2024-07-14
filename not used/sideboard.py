import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import streamlit_authenticator as stauth
import pickle

class sideboard:
	'''
	Class for a sidebard with the login screen
	'''
	def __init__(self, ):
		self.login_staus = False
		self.__build_page_layout()

	def __build_page_layout(self,):
		with st.sidebar:
			names = ['Chriss']
			usernames = ['Chriss']
			
			file_path = Path(__file__).parent / "hashed_pw.pkl"
			with file_path.open("rb") as file:
				hashed_passwords = pickle.load(file)
		
			authenticator = stauth.Authenticate(names,usernames, hashed_passwords,'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)
			
			name, authentication_status, username = authenticator.login("Login", "main")
			
			if authentication_status:
				authenticator.logout('Logout', 'main')
				st.write(f'Welcome *{name}*')
				secrets = True
			elif authentication_status == False:
				st.error('Username/password is incorrect')
				secrets = False
			elif authentication_status == None:
				st.warning('Please enter your username and password')
				secrets = False


	def is_loged_in(self):
		'''
		Method for returning the login status of the current user
		'''
		return self.login_status
