import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import streamlit as st
import time

class InputPage:
	'''
	Input page class for updating and including new results and decks
	in the database. Only for Users with an account
	'''
	def __init__(self, dm, df, hist_cols, cols):
		self.names = ['Chriss']
		self.usernames = ['Chriss']
		
		file_path = Path(__file__).parent / "hashed_pw.pkl"
		with file_path.open("rb") as file:
			self.hashed_passwords = pickle.load(file)
			
		self.__build_page_layout(df, hist_cols, cols, dm)

	def __build_page_layout(self, df, hist_cols, cols, dm):
		self.__login()
		if st.session_state['login']==True:
			self.__result_page(df, hist_cols, cols, dm)
		else:
			st.header('Bitte logge dich ein.')

	def __login(self):
		authenticator = stauth.Authenticate(self.names, self.usernames, self.hashed_passwords,
			'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)
		
		name, authentication_status, username = authenticator.login("Login", "main")
		
		if authentication_status:
			authenticator.logout('Logout', 'main')
			st.session_state['login'] = True
		elif authentication_status == False:
			st.error('Username/password is incorrect')
			st.session_state['login'] = False
		elif authentication_status == None:
			st.warning('Please enter your username and password')
			st.session_state['login'] = False
	
	def __result_page(self, df_elo, hist_cols, save_cols, dm):
		'''
		Method for creating a page for inserting and updating new decks and results to databse
		'''
		game, tour, deck, history, mod_stats = st.tabs(['Neues Spiel', 'Turniersieger', 'Neues Deck', 'Update', 'Deck modifizieren'])
		with game:
			self.__insert_new_game(dm, df_elo, hist_cols, save_cols)
		with tour:
			self.__insert_new_tournament_win(dm, df_elo, hist_cols, save_cols)
		with deck:
			self.__insert_new_deck(dm, df_elo, hist_cols, save_cols)
		with history:
			self.__create_elo_history(dm, df_elo, hist_cols, save_cols)
		with mod_stats:
			self.__modify_deck(dm, df_elo, hist_cols, save_cols)

	def __insert_new_game(self, dm, df_elo, hist_cols, save_cols):
		'''
		'''
		st.header('Trage neue Spielergebnisse ein: ')
		with st.form('Input new Games'):
			inputs = st.columns(6)
			with inputs[1]:
				deck1 = st.selectbox('Wähle Deck', 
							options=df_elo['Deck'].unique(),
							key='deck1')
				erg1 = st.number_input('Score ', 
										min_value=0,
										max_value=10000,
										step=1,
										key='erg2')
				
			with inputs[3]:
				deck2 = st.selectbox('Wähle Deck', 
							options=df_elo['Deck'].unique(),
							key='deck2')
				
				erg2 = st.number_input('Score ', 
										min_value=0,
										max_value=10000,
										step=1,
										key='erg1')
				
			with inputs[4]:
				if st.form_submit_button("Submit"):
						dm.update_elo_ratings(deck1, deck2, erg1, erg2, df_elo, hist_cols, save_cols)
						st.session_state['reload_flag'] = True
						st.success('Update erfolgreich!!')
						time.sleep(2)
						st.experimental_rerun()
					
	def __insert_new_tournament_win(self, dm, df_elo, hist_cols, save_cols):
		'''
		'''
		# insert a new tournament win
		st.header('Trage neuen Turniersieg ein:')
		with st.form('Tourniersieg'):
				inputs = st.columns(6)
				with inputs[1]:
					deck_tour = st.selectbox('Wähle Deck', 
								options=df_elo['Deck'].unique(),
								key='deck tournament')
				with inputs[2]:
					tournament = st.selectbox('Wähle Turnier:', 
												options=['Wanderpokal', 'Local','Fun Pokal'])
				with inputs[3]:
					result = st.selectbox('Ergebnis:', 
												options=['Teilnahme', 'Top','Win'])	
				with inputs[4]:
					if st.form_submit_button("Submit"):
						dm.update_tournament(deck_tour, tournament, result, df_elo, save_cols, hist_cols)
						st.session_state['reload_flag'] = True
						st.success('Update erfolgreich!!')
						time.sleep(2)
						st.experimental_rerun()

	def __insert_new_deck(self, dm, df_elo, hist_cols, save_cols):
		'''
		'''
		# insert a new deck
		st.header('Trage neues Deck ein:')
		with st.form('neies_deck'):
				inputs = st.columns(6)
				with inputs[1]:
					new_deck = st.text_input('Names des neuen Decks', key='new deck')
					owner = st.text_input('Names des Spielers', key='player')
					deck_type = st.selectbox('Wähle einen Decktype:',
												options=df_elo['Type'].unique(), key='decktype')
				with inputs[2]:
					attack = st.number_input('Attack-Rating', min_value=0, max_value=5, step=1, key='attack')
					control = st.number_input('Control-Rating', min_value=0, max_value=5, step=1, key='control')
					resilience = st.number_input('Resilience-Rating', min_value=0, max_value=5, step=1, key='resilience')
					
				with inputs[3]:
					recovery = st.number_input('Recovery-Rating', min_value=0, max_value=5, step=1, key='recovery')
					combo = st.number_input('Combo-Rating', min_value=0, max_value=5, step=1, key='combo')
					consistency = st.number_input('Consistency-Rating', min_value=0, max_value=5, step=1, key='consistency')
					
				with inputs[4]:
					if st.form_submit_button("Submit"):
						dm.insert_new_deck(new_deck, owner, attack, control, recovery, consistency, combo, resilience, deck_type)
						st.session_state['reload_flag'] = True
						st.success('Update erfolgreich!!')
						time.sleep(2)
						st.experimental_rerun()
				
	def __create_elo_history(self, dm, df_elo, hist_cols, save_cols):
		'''
		'''
		# set current elo in history by actual month and year	
		st.header('Update History:')
		with st.form('history_update'):
				inputs = st.columns(6)
				with inputs[4]:
					if st.form_submit_button("Submit"):
						dm.update_history(df_elo, hist_cols, save_cols)
						st.session_state['reload_flag'] = True
						st.success('Update erfolgreich!!')
						time.sleep(2)
						st.experimental_rerun()

	def __modify_deck(self, dm, df_elo, hist_cols, save_cols):
		'''
		'''
		# modify deck stats 				
		st.header('Modfiziere Deckstats:')
		with st.form('Mod Stats'):
				inputs = st.columns(6)
				with inputs[1]:
					deck_choose = st.selectbox('Wähle Deck', 
								options=df_elo['Deck'].unique(),
								key='deck_modify')
					
				with inputs[2]:
					in_stats = st.selectbox('Wähle Eigenschaft zum verändern:',
									options=['Attack', 'Control', 	'Recovery',
											'Consistensy',	 'Combo', 'Resilience'])
					modif_in = st.number_input('Rating:',
												min_value=0,
												max_value=5,
												step=1,
												key='type_modifier')
					
				with inputs[3]:
					in_stats_type = st.selectbox('Wähle Type:',
									options=['Type'])
					new_type = st.selectbox('Neuer Type:',
											options=df_elo['Type'].unique())
					
				with inputs[4]:
					if st.form_submit_button("Submit"):
						dm.update_stats(deck_choose, in_stats, modif_in, in_stats_type, new_type, df_elo, hist_cols, save_cols)
						st.session_state['reload_flag'] = True
						st.success('Update erfolgreich!!')
						time.sleep(2)
						st.experimental_rerun()
				