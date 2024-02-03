import numpy as np
import pandas as pd
import streamlit as st
import scipy.special as scsp
import itertools
import random
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pages._pages.Visualization import Visualization
import re
import requests
from PIL import Image
from io import BytesIO

class DeckBuilderPage:
    '''
    Class for a page to compare the results and some statistics of 
    some choosen decks.
    '''
    def __init__(self):
        '''
        Init-Method
        :param df: dataframe with elo data
        :param hist_cols: column names of past elo stats
        '''
        # Set dashboard colors
        self.back_color = "#d1e6f9"
        self.filed_color = "#0083B8"
        self.min_zoom = 3.5
        self.max_zoom = 8
        self.max_draws = 1000
        self.keep_cols = ['archetype', 'name', 'price', 'image', 'frameType']
        self.feat_cols = ['image', 'Karte', 'Anzahl', 'Tag', 'price', 'archetype', 'frameType']
        self.vis = Visualization()
        # build page
        if 'main_deck' not in st.session_state:
            self.__reset_session_state()
        self.ygo_db = self.__ygo_api()
        self.__build_page_layout()

    def __split_text_by_pattern(self, text:str)->list:
        """
        Method for extracting cards and amount of a big string
        Pattern 1x Stovie 2x Arianna
        :param text: string of all cards matching the pattern
        :return: list of matches patterns
        """
        # Verwendet regulären Ausdruck, um nach dem Muster "Number x Name" zu suchen
        pattern = re.compile(r"(\b\d+x+\s\b[a-zA-Z0-9,.:&\-' ]+)")
    
        # Findet alle Übereinstimmungen im Text
        return pattern.findall(text)
        

    @st.cache_data
    def __ygo_api(_self):
        """
        Api Call of the yugioh pro deck database of all cards
        :return ygo_db: DataFrame with feature of interes
        -----------------
        Calculated Feature:
        1. min price of cards
        2. image path of card
        """
        url = 'https://db.ygoprodeck.com/api/v7/cardinfo.php'
        respons = requests.get(url)
        ygo_db = pd.json_normalize(respons.json()['data'])
        ygo_db['price'] = ygo_db['card_prices'].apply(_self.get_price)
        ygo_db['image'] = ygo_db['card_images'].apply(_self.get_image_link)
        return ygo_db[_self.keep_cols]
    
    @st.cache_data
    def __fuzzy_search_ygo(_self, string:str) -> pd.DataFrame:
        # sourcery skip: simplify-empty-collection-comparison, simplify-str-len-comparison
        """
        Method for searching cards in pro deck datbase based on the fuzzy search endpoint
        :param string: string for searching
        :return cards: dataframe with matching cards of search
        """
        # return empty dataframe if string is empty
        if len(string)==0:
            return pd.DataFrame(columns=['image','Hinzufügen','name'])
        # get results of search
        url = f'https://db.ygoprodeck.com/api/v7/cardinfo.php?fname={string}'
        respons = requests.get(url)
        st.write(respons.status_code)
        cards = pd.json_normalize(respons.json()['data'])
        if cards.empty:
            return cards
        # calculate feature of search results
        cards['price'] = cards['card_prices'].apply(_self.get_price)
        cards['image'] = cards['card_images'].apply(_self.get_image_link)
        cards['Hinzufügen'] = False
        cards['Anzahl'] = 1
        cards['Tag'] = 'Rest'
        cards['Karte'] = cards['name']
        return cards[['Karte', 'Tag', 'name', 
                      'price', 'archetype', 'image', 
                      'Anzahl', 'Hinzufügen', 'frameType']]

    
    def get_image_link(self, x):
        """
        Method for extraction of the image link of the ygo pro deck
        api elment
        """
        return str(x[0]['image_url_cropped'])
    
    def get_price(self,x):
        """
        Method for getting the minial price of the ygo pro deck
        api element
        """
        y = []
        for idx in x:
            if 'cardmarket_price' in idx:
                y.append((float(idx['cardmarket_price'])))
            else:
                y.append(0)
        return min(y)

    def __create_dataframe(self, card_list:list) -> pd.DataFrame:
        """
        Method for creating a dataframe with all crads from the list input
        matched patterns
        :param card_list: list of strings with matched patterns of list input
        :return df: datframe with extracted feature of list input
        """
        # init features
        df = pd.DataFrame(columns=['Karte', 'Anzahl', 'Tag'])
        cards = []
        amounts = []
        # loop through data and extract need feature
        for card in card_list:
            splits = card.split('x ')
            amounts.append(int(splits[0]))
            cards.append(''.join(splits[1:]).strip())
        # set feature to dataframe
        df['Karte'] = cards
        df['Anzahl'] = amounts
        df['Tag'] = 'Rest'
        return df

    def get_first_image(self, x:pd.Series) -> str:
        """
        Method for extracting the first given valid image
        link of a list of possible links
        :param x: list/Series of links
        :return item: valid link to image
        """
        for item in x:
            if not item:
                continue

            return item

    def __build_page_layout(self,):
        """
        Layout of the deck building page
        Structured in 3 tabs
        Tab 1: Input
        Tab 2: Deck Breakdown
        Tab 3: Statistical Analysis
        """
        st.header('Deck Builder')
        tabs = st.tabs(['Eingabe', 'Deck Breakdown', 'Deck Analyse'])
        self.__create_input_tab(tabs[0])
        self.__create_breakdown_tab(tabs[1])
        self.__create_analysis_tab(tabs[2])

    def __create_input_tab(self, st_obj, ):
        """
        Layout Method for the first tab (Input Tab)
        Structured in 3 columns
        1st Col: List input of deck parts
        2nd Col: Tagging of cards
        3rd Col: Seach for some cards 
        :param st_obj: streamlit tab object for placing in the right tab
        """
        cols = st_obj.columns([.8,2,1.2])
        self.__list_input(cols[0])
        self.__tagging_editor(cols[1])
        self.__searcher_input(cols[2])

    def __searcher_input(self, st_obj):
        """
        Method for handling the Search field and card selection for input tab
        param st_obj: streamlit position object
        """
        # define search field
        text = st_obj.text_input('Suche Karte:', placeholder='Test')
        df = self.__fuzzy_search_ygo(text)
        # define dataeditor for displaying search results
        form = st_obj.form('searcher')
        df_select = form.data_editor(df[['image','Hinzufügen','name' ]], 
                                    hide_index=True,
                                    height = 500,
                                    column_config={ "image": st.column_config.ImageColumn("", width='small')})
        # submit selected search results to card tagging
        cols = form.columns(3)
        if cols[0].form_submit_button('Zum Main'):
            self.__add_selection_to_session_state(df_select, df, 'main')
        if cols[1].form_submit_button('Zum Side'):
            self.__add_selection_to_session_state(df_select, df, 'side')
        if cols[2].form_submit_button('Zum Extra'):
            self.__add_selection_to_session_state(df_select, df, 'extra')

    def __add_selection_to_session_state(self, df_select:pd.DataFrame, df:pd.DataFrame, option:str):
        """
        Method for adding selected searched cards to tagging and analysis dataframes
        :param df_selected: dataframe with cards for adding
        :param df: dataframe with all seached cards
        :param option: part of deck for adding
        """
        # get all feature of the choosen cards
        idx = df[df_select['Hinzufügen']].index
        if idx.empty:
            return
        # merge selection to session state deck dataframe
        st.session_state[f'{option}_deck'] = self.merge_sst(st.session_state[f'{option}_deck'],
                                                            df.loc[idx])
        st.experimental_rerun()

    def __merge_into_session_state(self, df_main, df_side, df_extra):
        """
        Method for adding/merging new cards to deck dataframe
        :param df_main: ain decj dataframe
        :param df_side: side deck dataframe
        :param df_extra: extra deck dataframe
        """
        # init session state if not existing
        if 'main_deck' not in st.session_state:
            self.__reset_session_state()
        # merge dataframes
        if not df_main.empty:
            st.session_state['main_deck'] = self.merge_sst(st.session_state['main_deck'], df_main)
        if not df_side.empty:
            st.session_state['side_deck'] = self.merge_sst(st.session_state['side_deck'], df_side)
        if not df_extra.empty:
            st.session_state['extra_deck'] = self.merge_sst(st.session_state['extra_deck'], df_extra)

    def merge_sst(self, df1, df2):
        """
        Add Cards to datafme that are not in the dataframe
        :param df1: dataframe into merging new cards
        :param df2: dataframe with cards for merging
        """
        idx = df2[~df2['Karte'].isin(df1['Karte'].to_list())].index
        cols = list(df1.columns)
        return df1 if idx.empty else pd.concat([df1, df2.loc[idx, cols]], ignore_index=True)
            
    def __list_input(self, st_obj):
        """
        Method for creating and handling the list input for easy deck tagging
        :param st_obj: streamlit object for placing elements
        """
        # define list input layout with 3 tabs for each deck type
        input_form = st_obj.form('Input')
        input_tabs = input_form.tabs(['Main-Deck', 'Extra-Deck', 'Side-Deck'])
        placeholder='''1x Left Arm of the Forbidden One'''
        main_str = input_tabs[0].text_area('Main-Deck Liste:', height=480, placeholder=placeholder)
        extra_str = input_tabs[1].text_area('Extra-Deck Liste:', height=480, placeholder=placeholder)
        side_str = input_tabs[2].text_area('Side-Deck Liste:', height=480, placeholder=placeholder)
        # handling data extracktion if input is submitted
        if input_form.form_submit_button('Übertrage Decks'):
            # main 
            df_main = self.__create_table_from_decklist(main_str)
            # extra
            df_extra = self.__create_table_from_decklist(extra_str)
            # side
            df_side = self.__create_table_from_decklist(side_str)
            # merge decks into session state
            self.__merge_into_session_state(df_main, df_side, df_extra)
        
        
    def __create_table_from_decklist(self, string):
        """
        Method for creating a dataframe with all feature of list input data
        :param string: input string with list of cards
        :return datframe: dataframe with card feature 
        """
        # return empty dataframe if list is empty
        if not string:
            return pd.DataFrame(columns=self.feat_cols)
        # find patterns and create dataframe
        matches = self.__split_text_by_pattern(string)
        df = self.__create_dataframe(matches)
        # merge ygo database data to extracted cards
        return pd.merge(df, self.ygo_db, left_on='Karte', right_on='name', how='left')

    def __reset_session_state(self,):
        """
        Method for intialise or reset session state deck data
        """
        st.session_state['main_deck'] = pd.DataFrame(columns=self.feat_cols)
        st.session_state['side_deck'] = pd.DataFrame(columns=self.feat_cols)
        st.session_state['extra_deck'] = pd.DataFrame(columns=self.feat_cols)

    def __tagging_editor(self, st_obj):
        """
        Method for tagging cards for further deck analysis
        :param st_obj: streamlit object for placing
        """
        cols = st_obj.columns([2,2,1])
        # creating tagging options
        deck_ops = ['Starter', 'Brick', 'Extender', 'Handtrap', 'Boardbreaker', 'Engine', 'Non-Engine']
        df_ops = cols[0].expander('Tag Auswahl', expanded=False).data_editor(pd.DataFrame(deck_ops, columns=['Tags']), 
                                                             num_rows='dynamic')  
        # button for deck reset        
        cols[2].button('Rest', on_click=self.__reset_session_state, use_container_width=True)
        # layout deck part tagging forms
        tagging_form = st_obj.form(key='Inputform')
        tagging_tabs = tagging_form.tabs(['Main-Deck', 'Extra-Deck', 'Side-Deck'])
        # tagging main deck
        df_main = tagging_tabs[0].data_editor(st.session_state['main_deck'], 
                        column_config={
                            "Tag": st.column_config.SelectboxColumn(
                                "Tag",help="Rolle der Karte im Deck",
                                width="medium",options=df_ops['Tags'].to_list(),required=True),
                            "image": st.column_config.ImageColumn("", width='small')
            },hide_index=True, num_rows='dynamic', key='maindeck')
        # tagging extra deck
        df_extra = tagging_tabs[1].data_editor(st.session_state['extra_deck'], 
                        column_config={
                            "Tag": st.column_config.SelectboxColumn(
                                "Tag",help="Rolle der Karte im Deck",
                                width="medium",options=df_ops['Tags'].to_list(),required=True),
                            "image": st.column_config.ImageColumn("", width='small')
            },hide_index=True, num_rows='dynamic', key='extradeck')
        # tagging side deck
        df_side = tagging_tabs[2].data_editor(st.session_state['side_deck'], 
                        column_config={
                            "Tag": st.column_config.SelectboxColumn(
                                "Tag",help="Rolle der Karte im Deck",
                                width="medium",options=df_ops['Tags'].to_list(),required=True),
                            "image": st.column_config.ImageColumn("", width='small')
            },hide_index=True, num_rows='dynamic', key='sidedeck')
        # set tagged datframes into session state if data get submitted
        if tagging_form.form_submit_button('Speichere/Berechne Deck'):
            st.session_state['main_deck'] = df_main
            st.session_state['side_deck'] = df_side
            st.session_state['extra_deck'] = df_extra
        
    def __create_breakdown_tab(self, tab):
        """
        Method for creating the deck breackdown tab of the deck analysis
        :param tab: stramlit pacing object
        """
        # breakdown tab
        breakdown_cols = tab.columns(6)
        # calculate deck part price brackdown as metric
        mainprice = self.__calculate_total_price(st.session_state['main_deck'])
        extraprice = self.__calculate_total_price(st.session_state['extra_deck'])
        sideprice = self.__calculate_total_price(st.session_state['side_deck'])
        breakdown_cols[0].metric('Gesamtpreis:', f'{np.round(mainprice+extraprice+sideprice,2)}€')
        breakdown_cols[1].metric('Preis Main Deck:', f'{mainprice}€')
        breakdown_cols[2].metric('Preis Extra Deck:', f'{extraprice}€')
        breakdown_cols[3].metric('Preis Side Deck:', f'{sideprice}€')
        # plot deck rations
        plot_cols = tab.columns([1,.7,1])
        # Archetype: feature claculation and plotting
        df = st.session_state['main_deck'].groupby('archetype')['Anzahl'].sum()
        if not df.empty:
            df = df.reset_index()
            plot_cols[0].plotly_chart(self.vis.ploty_bar(df, 'archetype', 'Anzahl', title='Archetypes'),
                                  use_container_width=True)
        # Card Ratio: feature claculation and plotting
        st.session_state['main_deck']['frameType'].fillna('effect', inplace=True)
        df = st.session_state['main_deck'].groupby('frameType')['Anzahl'].sum().reset_index()
        df = self.reset_values(df, 'frameType', 'effect', 'Monster')
        df = self.reset_values(df, 'frameType', 'spell', 'Zauber')
        df = self.reset_values(df, 'frameType', 'trap', 'Falle')
        df, color_dict= self.sort_and_get_colors(df, 'frameType')
        color = list(color_dict.keys())
        if not df.empty:
            plot_cols[1].plotly_chart(self.vis.plotly_pie(df, 'Anzahl', 'frameType', title='Kartentyp',
                                                      color=color,
                                                      color_dict=color_dict),
                                    use_container_width=True)
        # Extra Deck Ratio: feature claculation and plotting
        df = st.session_state['extra_deck'].groupby('frameType')['Anzahl'].sum().reset_index()
        if not df.empty:
            df['frameType'] = df['frameType'].str.title()
            plot_cols[2].plotly_chart(self.vis.ploty_bar(df, 'frameType', 'Anzahl', True, 'Extra-Deck'),
                                  use_container_width=True)


    def sort_and_get_colors(self, df, col):
        """
        Helper Method for plotting card types with right color
        :param df: dataframe with aggregated card type feature
        :param col: name of feature column
        :return dataframe in right order of feature for plotting
        """
        tmp = []
        color = {'Zauber':'green', 'Falle':'red', 'Monster':'orange'}
        dict_return = {}
        for string in color:
            idx = df[df[col] == string].index
            tmp.append(df.loc[idx])
            if not idx.empty:
                dict_return[string] = color[string]
        return pd.concat(tmp, ignore_index=True), dict_return
    
    def reset_values(self, df, col, old, new):
        """
        Method for reset all values in ca column with an new value
        :param df: datframe with data to reset
        :param col: column with data for reset
        :param old: value for reseting
        :param new: new value
        :return df: dataframe with resetted values
        """
        idx = df[df[col]==old].index
        df.loc[idx, col] = new
        return df

    def __create_analysis_tab(self, tab):
        """
        Method for creating and handling the statistical analysis tab of the page
        :param tab: streamlit object for placing
        """
        # dont build tab if main deck is empty
        if st.session_state['main_deck'].empty:
            return
        # calculate stats from input
        deck_size = int(st.session_state['main_deck']['Anzahl'].sum())
        amount_of_tags = st.session_state['main_deck']['Tag'].nunique() #len(Ne)
        counts_per_tag = st.session_state['main_deck'].groupby('Tag')['Anzahl'].sum().astype(int) #Ne
        tag_list = counts_per_tag.index.to_list()
        image_list = st.session_state['main_deck'].groupby('Tag')['image'].apply(self.get_first_image)
        # analyse tab
        most_probable_comb, wsk, n_combis = self.__find_most_probable_combination(deck_size, 
                                                                                  amount_of_tags,
                                                                                  5, counts_per_tag)
        
        # metrics
        metrics = tab.columns(len(tag_list)+1)
        metrics[0].container(border=True).metric('Deckgröße', deck_size)
        for idx, tag in enumerate(tag_list):
            metrics[idx+1].container(border=True).metric(tag, 
                                                         int(counts_per_tag[tag]), 
                                                         f'{np.round(100*counts_per_tag[tag]/deck_size,2)}%')
        # figures
        progbar = tab.progress(0, 'Berechne Wahrscheinlichkeitentabelle')
        fig_cols = tab.columns([1.5 ,1, 1])
        # header
        fig_cols[0].markdown('### Wahrscheinlichkeiten von Kartentypen auf der Starthand in %')
        fig_cols[1].markdown('### Wahrscheinlichste Hand')
        fig_cols[1].text(f'Wahrscheinlichkeit: {int(wsk*10000)/100}%')
        fig_cols[1].text(f'{n_combis} Kombinationen')
        fig_cols[2].markdown('### Häufigste Hand')
        fig_cols[2].text('Durschnittliche Anzahl Karten pro Klasse nach 1000 Starthänden')
        # wsk table
        going_first = self.__build_start_hand(deck_size, counts_per_tag, tag_list)
        fig_cols[0].dataframe(going_first.round(2), hide_index=True)
        # wsk hand
        progbar.progress(33, 'Berechne wahrscheinlichste Hand')
        num_cards = []
        group = []
        for idx in range(amount_of_tags):
            num_cards.append(most_probable_comb.count(idx))
            group.append(tag_list[idx])
        fig = self.create_pie_chart(num_cards, group, image_list)
        fig_cols[1].pyplot(fig, tansparent=True, use_container_width=True)
        # frequent hand
        progbar.progress(66, 'Berechne Häufigste Hand')
        df = self.__get_most_frequent_hand(counts_per_tag, tag_list)
        fig = self.create_pie_chart(df['mittlere Anzahl'].to_list(), df['Kartentyp'].to_list(), image_list)
        fig_cols[2].pyplot(fig, tansparent=True, use_container_width=True)
        progbar.empty()
        
    def __calculate_total_price(self, df):
        """
        Method for calculationg the total price of a deck
        :param df: dataframe with prices and number of cards for price calculation
        """
        return np.round((df['Anzahl']*df['price']).sum(),2)
    
    def __get_most_frequent_hand(self, M, groups):
        """
        Method for calculation of the most frequent hand for the current deck
        using a max draw draws from the tagg distribution
        :param M: dataframe with number of cards for each tag
        :param groups: list of possible tags
        :return dataframe: results from draws
        """
        # init result data and deck array
        df = pd.DataFrame(index=range(self.max_draws), columns=groups)
        a = [(i+1)*np.ones(num) for i, num in enumerate(M)]
        a = np.concatenate(a)
        # draw n times and count drawn tags
        for number in range(self.max_draws): 
            random.shuffle(a)
            for idx in range(len(groups)):
                count = list(a[:5]).count(idx+1)
                df.at[number, groups[idx]] = count
        return df.mean().reset_index().rename(columns={0:'mittlere Anzahl', 'index':'Kartentyp'})


    def __build_start_hand(self, deck_size, card_amounts, groups):
        """
        Method for calculating the probabilty of the min amount of cards on the
        starting hand for the cards of all choosen tags
        :param deck_size: numberof cards in the deck
        :param card_amounts: amount of cards of each tag
        :param groups: choosen tags
        """
        # define result variables for saving
        df_first = pd.DataFrame(columns=groups, index=range(6))
        df_first['Anzahl Karten'] = ['kein', 'min. 1', 'min. 2', 'min. 3', 'min. 4', 'min. 5']
        # calculate probabilties using combinatoric approach
        for cat, amount in zip(groups, card_amounts):
            df_first[cat] = self.__calc_min_hand(deck_size, amount, 5)
        df_first[groups] = (df_first[groups]*100)
        df_first = df_first.reindex(sorted(df_first.columns), axis=1)
        return df_first

    def __calc_min_hand(self,deck_size, amount_cards, draws):
        """
        Mehtod for calcualting the probabilty by the given card tag distribution
        for n drwn cards
        :param deck_size: numberof cards in the deck
        :param card_amounts: amount of cards of each tag
        :param drws: number of drawn cards 5 for strating hand
        """
        wsk = np.zeros(draws+1)
        tmp = np.zeros(draws+1)
        for i in range(draws+1):
            tmp[i] = self.__calc_wsk_exact(deck_size, amount_cards, draws, i)
            if i==0:
                wsk[0] = tmp[0] 
            else:
                wsk[i] = 1-np.sum(tmp[:i])
        return wsk


    def __calc_wsk_exact(self, deck_size, amount_cards, hand_size, x):
        """
        Method for calculation the probabilty for min. x cards in a hand of handsize
        cards out of a deck with a given size using a binomial cofficient approach from
        combinatoric
        :param deck_size: numberof cards in the deck
        :param card_amounts: amount of cards of each tag
        :param hand_size: number of drawn crads from deck
        :param x: min number of drwawn cards of choosen type
        :return probaility: prop of min cards per typ
        """
        return scsp.binom(amount_cards, x)*scsp.binom(deck_size-amount_cards, hand_size-x)/scsp.binom(deck_size, hand_size)
    
    
    def __find_most_probable_combination(self, N, M, k, amount, hand_size=5):
        """
        Finds the most probable combination of events after k draws from M possible event groups.

        Parameters:
        N (int): Total number of particles.
        M (int): Number of possible event groups.
        k (int): Number of draws.
        amount (list): List of length M containing the probabilities of each event group.

        Returns:
        tuple: Tuple containing the most probable combination and its probability.
        """

        events_combinations = itertools.combinations_with_replacement(range(M), k)
        max_probability = 0.0
        most_probable_combination = ()
        combis = 0
        for combination in events_combinations:
            combis += 1
            prob_product = 1.0
            for group_idx in range(M):
                count = combination.count(group_idx)
                
                prob = self.__calc_wsk_exact(N, amount[group_idx], hand_size, count)
                prob_product *= prob

            if prob_product > max_probability:
                max_probability = prob_product
                most_probable_combination = combination

        return most_probable_combination, max_probability, combis
    
    def get_image_from_ygo(self, url):
        """
        Method for getting the image for displaying from a link
        :param url: link to the image
        :return array: array with the image data
        """
        res = requests.get(url)
        if res.status_code == 200:
            img_arr = Image.open(BytesIO(res.content))
            img_arr.thumbnail((256, 256))
        return np.array(img_arr)
        
    def create_pie_chart(self, total, labels, links):
        """
        Method for creating a pie cards with images in the background of the 
        choosen part using matplotlib
        :param total: values for displaying in the plot
        :param label: label of the value classes
        :param links: links to the class images
        """
        fig, axs = plt.subplots(1, figsize=(30, 30), facecolor=self.back_color)
        axs.axis("equal")
        # modify labels
        tmp = pd.DataFrame(columns=['values', 'labels'])
        tmp['values'] = total
        tmp['labels'] = labels
        tmp['ex'] = .05
        tmp = tmp[tmp['values']>0]
        tmp.sort_values(by='values', ascending=False, inplace=True)
        tmp.reset_index(drop=True, inplace=True)
        total = tmp['values'].to_list()
        labels = tmp['labels'].to_list()
        labs = [f'{lab}\n{tot}' for tot, lab in zip(total, labels)]
        
        # create pychart
        wedges, texts = axs.pie(total, startangle=90, labels=labs, explode=tmp['ex'].to_list(),
                            wedgeprops = { 'linewidth': 2, "edgecolor" :"white","fill":False,},
                            textprops={'fontsize': 55, 'color':'black', })
        # set background images
        sum_total = np.sum(total)
        counter = 0
        for i in range(len(labels)):
            category_percentage = total[i]/sum_total
            counter += category_percentage
            if labels[i] == 'Rest':
                fn = f"./Deck_Icons/{labels[i]}.png"
                im = plt.imread(fn, format='png')
            else:
                fn = links[labels[i]]
                im = self.get_image_from_ygo(fn)

            x = 0.65*np.cos(2*np.pi*counter + np.pi/4)
            y = 0.65*np.sin(2*np.pi*counter + np.pi/4)
            zoom = self.min_zoom*category_percentage/.20
            zoom = np.max([self.min_zoom, zoom])
            zoom = np.min([self.max_zoom, zoom])
            self.img_to_pie(im, wedges[i], xy=(x,y), zoom=zoom)
            wedges[i].set_zorder(10)
            
        return fig

    def img_to_pie(self, im, wedge, xy, zoom=1, ax = None):
        """
        Method for adding the image to the pie chart
        """
        if ax is None: ax=plt.gca()

        path = wedge.get_path()
        patch = PathPatch(path, facecolor='none')
        ax.add_patch(patch)
        imagebox = OffsetImage(im, zoom=zoom, clip_path=patch, zorder=-10)
        ab = AnnotationBbox(imagebox, xy, xycoords='data', pad=0, frameon=False)
        ax.add_artist(ab)
