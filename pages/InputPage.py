import streamlit_authenticator as stauth
import streamlit as st
import time
from DataModel.DeckModel import DeckModel
from DataModel.TournamentModel import TournamentModel

class InputPage:
	'''
	Input page class for updating and including new results and decks
	in the database. Only for Users with an account
	'''
	def __init__(self):
		self.deck_model = DeckModel(load_data=False)
		self.tourn_model = TournamentModel(load_data=False)

		#self.__login()
		self.__build_page_layout(st.session_state['deck_data'], 
						   st.session_state['history_columns'], 
						   st.session_state['columns'])

	def __build_page_layout(self, df, hist_cols, cols):
		#if st.session_state['login']==True:
		self.__result_page(df, hist_cols, cols)
		#else:
			#st.header('Bitte logge dich ein.')
			#self.__login()

	def __login(self):
		creds = {'usernames':{}}
		for u,p in zip(st.secrets['users'], st.secrets['pwd']):
			creds['usernames'][u] = {'name':u, 'password':p}
		
		authenticator = stauth.Authenticate({'credintials':creds, 'cookie':{}})

		name, authentication_status, username = authenticator.login(location='sidebar')

		if authentication_status:
			authenticator.logout('Logout', 'main')
			st.session_state['login'] = True
		elif authentication_status == False:
			st.error('Username/password is incorrect')
			st.session_state['login'] = False
		elif authentication_status is None:
			st.warning('Please enter your username and password')
			st.session_state['login'] = False
	
	def __result_page(self, df_elo, hist_cols, save_cols):
		'''
		Method for creating a page for inserting and updating new decks and results to databse
		'''
		game, tour, deck, history, mod_stats = st.tabs(['Neues Spiel', 'Turniere', 'Neues Deck', 'Update', 'Deck modifizieren'])
		with game:
			self.__insert_new_game(self.deck_model, df_elo, hist_cols, save_cols)
		with tour:
			self.__insert_new_tournament_win(self.tourn_model, df_elo)
		with deck:
			self.__insert_new_deck(self.deck_model, df_elo, hist_cols, save_cols)
		with history:
			self.__create_elo_history(self.deck_model, df_elo, hist_cols, save_cols)
		with mod_stats:
			self.__modify_deck(self.deck_model, df_elo, hist_cols, save_cols)

	def __insert_new_game(self, dm:DeckModel, df_elo, hist_cols, save_cols):
		'''
		'''
		st.header('Trage neue Spielergebnisse ein: ')
		form = st.form('Input new Games')
		inputs = form.columns(6)
		# deck and result 1
		deck1 = inputs[1].selectbox('Wähle Deck', 
					options=df_elo['Deck'].unique(),
					key='deck1')
		erg1 = inputs[1].number_input('Score ', 
								min_value=0,
								max_value=10000,
								step=1,
								key='erg1')
				
		# deck and result 2
		deck2 = inputs[3].selectbox('Wähle Deck', 
					options=df_elo['Deck'].unique(),
					key='deck2')
		
		erg2 = inputs[3].number_input('Score ', 
								min_value=0,
								max_value=10000,
								step=1,
								key='erg2')
			
		if form.form_submit_button("Submit"):
				dm.update_elo_ratings(deck1, deck2, erg1, erg2, df_elo, hist_cols, save_cols)
				self.__after_update(inputs[4])
					
	def __insert_new_tournament_win(self, dm:TournamentModel, df_elo):
		'''
		'''
		# insert a new tournament win
		st.header('Trage neues Turnier ein:')
		form = st.form('Turniere:')
		inputs = form.columns(6)
		# choose deck 1
		result = {'Deck':inputs[1].selectbox('Wähle Deck', 
					options=df_elo['Deck'].unique(),
					key='deck tournament')}
		result['Win'] = inputs[1].number_input('Wins', min_value=0, step=1)
		# choose tournament
		result['Mode'] = inputs[2].selectbox('Wähle Turnier:', 
									options=['Wanderpokal', 'Local','Fun Pokal', 'Regional'])
		result['Draw'] = inputs[2].number_input('Draws', min_value=0, step=1)
		# give result
		result['Standing'] = inputs[3].selectbox('Ergebnis:', 
									options=['Teilnahme', 'Top','Win'])
		result['Loss'] = inputs[3].number_input('Losses', min_value=0, step=1)	
		# get date
		result['Date'] = inputs[4].text_input('Datum:', max_chars=10, placeholder='1970-01-01')
		
		if form.form_submit_button("Submit"):
			dm.insert_tournament(result)
			self.__after_update(inputs[4])

	def __insert_new_deck(self, dm:DeckModel, df_elo, hist_cols, save_cols):
		'''
		'''
		# insert a new deck
		st.header('Trage neues Deck ein:')
		form = st.form('neies_deck')
		inputs = form.columns(6)
		# deck, player and type
		new_deck = inputs[1].text_input('Names des neuen Decks', key='new deck')
		owner = inputs[1].text_input('Names des Spielers', key='player')
		deck_type = inputs[1].selectbox('Wähle einen Decktype:',
									options=df_elo['Type'].unique(), key='decktype')
		# attack, control and resiliance
		attack = inputs[2].number_input('Attack-Rating', min_value=0, max_value=5, step=1, key='attack')
		control = inputs[2].number_input('Control-Rating', min_value=0, max_value=5, step=1, key='control')
		resilience = inputs[2].number_input('Resilience-Rating', min_value=0, max_value=5, step=1, key='resilience')
					
		# recovery, combo and consistency
		recovery = inputs[3].number_input('Recovery-Rating', min_value=0, max_value=5, step=1, key='recovery')
		combo = inputs[3].number_input('Combo-Rating', min_value=0, max_value=5, step=1, key='combo')
		consistency = inputs[3].number_input('Consistency-Rating', min_value=0, max_value=5, step=1, key='consistency')
					
		if form.form_submit_button("Submit"):
			dm.insert_new_deck(new_deck, owner, attack, control, recovery, consistency, combo, resilience, deck_type)
			self.__after_update(inputs[4])
				
	def __create_elo_history(self, dm:DeckModel, df_elo, hist_cols, save_cols):
		'''
		'''
		# set current elo in history by actual month and year	
		st.header('Update History:')
		form =  st.form('history_update')
		inputs = form.columns(6)		
		if form.form_submit_button("Submit"):
			dm.update_history(df_elo, hist_cols, save_cols)
			self.__after_update(inputs[4])

	def __modify_deck(self, dm:DeckModel, df_elo, hist_cols, save_cols):
		'''
		'''
		# modify deck stats 				
		st.header('Modfiziere Deckstats:')
		form = st.form('Mod Stats')
		inputs = form.columns(6)
		# choosen deck
		deck_choose = inputs[1].selectbox('Wähle Deck', 
								options=df_elo['Deck'].unique(),
								key='deck_modify')
					
		# stats and values
		in_stats = inputs[2].selectbox('Wähle Eigenschaft zum verändern:',
						options=['Attack', 'Control', 	'Recovery',
								'Consistensy',	 'Combo', 'Resilience'])
		modif_in = inputs[2].number_input('Rating:',
									min_value=0,
									max_value=5,
									step=1,
									key='type_modifier')
					
		# type and value
		in_stats_type = inputs[3].selectbox('Wähle Type:',
						options=['Type'])
		new_type = inputs[3].selectbox('Neuer Type:',
								options=df_elo['Type'].unique())
		
		# submit
		if form.form_submit_button("Submit"):
			dm.update_stats(deck_choose, in_stats, modif_in, in_stats_type, new_type, df_elo, hist_cols, save_cols)
			self.__after_update(inputs[3])			

	def __after_update(self, pos):
		st.session_state['reload_flag'] = True
		pos.success('Update erfolgreich!!')
		time.sleep(2)
		st.experimental_rerun()
				