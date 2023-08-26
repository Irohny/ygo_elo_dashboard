import numpy as np
import pandas as pd
import streamlit as st
import scipy.special as scsp
import itertools
import random
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
        self.back_color = "#00172B"
        self.filed_color = "#0083B8"
        self.min_zoom = 3.5
        self.max_zoom = 8
        # build page
        self.__build_page_layout()

    def __build_page_layout(self,):
        """
        """
        with st.form('PreSelection'):
            cols = st.columns([1,2])
            deck_size = cols[0].number_input(label='Anzahl Karten im Deck', min_value=40, max_value=60, value=40)
            groups = cols[1].multiselect('Wähle Kartenkategorien:', options=list(st.session_state['card_amounts'].keys()), 
                                         default=st.session_state['cardtyp_defaults'])
            cols[0].form_submit_button('Bestätige die Auswahl')

        with st.form('Deck Ratio'):
            cols_amount = st.columns([1,1,1,1,1,1])
            n_rows = len(groups)//6+1
            counter = 0
            selected_amounts = {}
            for i in range(6):
                for r in range(n_rows):
                    if counter>=len(groups):
                        break
                    selected_amounts[groups[counter]] = cols_amount[i].number_input(label=f'Anzahl {groups[counter]}', 
                                                                                          min_value=0, 
                                                                                          max_value=deck_size, 
                                                                                          value=st.session_state['card_amounts'][groups[counter]]) 
                    counter += 1
                    

            cols_amount[0].form_submit_button('Ergebnis')
        
        # get selected group cards
        selected_groups = list(selected_amounts.keys())
        Ne = list(selected_amounts.values())
        for key in list(st.session_state['card_amounts'].keys()):
            if key in selected_groups:
                st.session_state['card_amounts'][key] = selected_amounts[key]
            else:
                st.session_state['card_amounts'][key] = 0

        total_sum = sum(Ne)
        print_flag = True
        #
        if total_sum<deck_size:
            Ne.append(deck_size-total_sum)
            selected_groups.append('Rest')
        elif total_sum>deck_size or total_sum==0:
            print_flag = False
            st.success('Die Anzahl der Karten in den Klassen ist größer als die Anzahl der Karten im Deck')

        
        if print_flag:
            figures, tables = st.tabs(['Handkarten', 'Wahrscheinlichkeiten'])       
            #
            with figures:
                cols = st.columns([.5, 2,2,.5])
                most_probable_comb, wsk, n_combis = self.__find_most_probable_combination(deck_size, len(Ne), 5, Ne)
                cols[1].markdown('### Wahrscheinlichste Hand')
                cols[1].text(f'Wahrscheinlichkeit: {int(wsk*10000)/100}% bei {n_combis} Kombinationen')
                num_cards = []
                group = []
                for idx in range(len(Ne)):
                    num_cards.append(most_probable_comb.count(idx))
                    group.append(selected_groups[idx])
                fig = self.create_pie_chart(num_cards, group)
                cols[1].pyplot(fig, tansparent=True)
                
                cols[2].markdown('### Häufigste Hand')
                cols[2].text(f'Durschnittliche Anzahl Karten pro Klasse nach 1000 Starthänden')
                df = self.__get_most_frequent_hand(Ne, selected_groups)
                fig = self.create_pie_chart(df['mittlere Anzahl'].to_list(), df['Kartentyp'].to_list())
                cols[2].pyplot(fig, tansparent=True)

            #
            with tables:
                going_first = self.__build_start_hand(deck_size, Ne, selected_groups)
                st.markdown('### Wahrscheinlichkeiten von Kartentypen auf der Starthand in %')
                st.table(going_first)
                
    def __get_most_frequent_hand(self, M, groups):
        """
        """
        a = []
        df = pd.DataFrame(index=range(1000), columns=groups)
        for i, num in enumerate(M):
            a.append((i+1)*np.ones(num))
        a = np.concatenate(a)
        
        for number in range(1000): 
            random.shuffle(a)

            for idx in range(len(groups)):
                count = list(a[:5]).count(idx+1)
                df.at[number, groups[idx]] = count
        return df.mean().reset_index().rename(columns={0:'mittlere Anzahl', 'index':'Kartentyp'})


    def __build_start_hand(self, deck_size, card_amounts, groups):
        """
        """
        # first
        df_first = pd.DataFrame(columns=groups, index=range(6))
        df_first['Anzahl Karten'] = ['kein', 'min. 1', 'min. 2', 'min. 3', 'min. 4', 'min. 5']

        for cat, amount in zip(groups, card_amounts):
            df_first[cat] = self.__calc_min_hand(deck_size, amount, 5)
        df_first[groups] = (df_first[groups]*100)
        df_first = df_first.reindex(sorted(df_first.columns), axis=1)
        return df_first

    def __calc_min_hand(self,deck_size, amount_cards, draws):
        """
        """
        wsk = np.zeros(draws+1)
        tmp = np.zeros(draws+1)
        for i in range(0, draws+1):
            tmp[i] = self.__calc_wsk_exact(deck_size, amount_cards, draws, i)
            if i==0:
                wsk[0] = tmp[0] 
            else:
                wsk[i] = 1-np.sum(tmp[0:i])
        return wsk


    def __calc_wsk_exact(self, deck_size, amount_cards, hand_size, x):
        wsk = scsp.binom(amount_cards, x)*scsp.binom(deck_size-amount_cards, hand_size-x)/scsp.binom(deck_size, hand_size)
        return wsk
    
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
    
    
    def create_pie_chart(self, total, labels):
        fig, axs = plt.subplots(1, figsize=(15, 15), facecolor=self.back_color)
        axs.axis("equal")
        # modify labels
        labs = []
        values_to_remove = []
        labels_to_remove = []
        for tot, lab in zip(total, labels):
            if tot <=0:
                values_to_remove.append(tot)
                labels_to_remove.append(lab)
                continue
            labs.append(f'{lab}\n{tot}')
        # remove invalid values and labels from list
        for re_values, re_lab in zip(values_to_remove, labels_to_remove):
            total.remove(re_values)
            labels.remove(re_lab)
        # create pychart
        wedges, texts = axs.pie(total, startangle=90, labels=labs,
                            wedgeprops = { 'linewidth': 2, "edgecolor" :"white","fill":False,},
                            textprops={'fontsize': 25, 'color':'white'})
        # set background images
        sum_total = np.sum(total)
        counter = 0
        for i in range(len(labels)):
            category_percentage = total[i]/sum_total
            counter += category_percentage
            fn = f"./Deck_Icons/{labels[i]}.png"
            x = 0.5*np.cos(2*np.pi*counter + np.pi/4)
            y = 0.5*np.sin(2*np.pi*counter + np.pi/4)
            zoom = self.min_zoom*category_percentage/.20
            zoom = np.max([self.min_zoom, zoom])
            zoom = np.min([self.max_zoom, zoom])
            self.img_to_pie(fn, wedges[i], xy=(x,y), zoom=zoom)
            wedges[i].set_zorder(10)
            
        return fig

    def img_to_pie(self, fn, wedge, xy, zoom=1, ax = None):
        if ax==None: ax=plt.gca()
        im = plt.imread(fn, format='png')
        path = wedge.get_path()
        patch = PathPatch(path, facecolor='none')
        ax.add_patch(patch)
        imagebox = OffsetImage(im, zoom=zoom, clip_path=patch, zorder=-10)
        ab = AnnotationBbox(imagebox, xy, xycoords='data', pad=0, frameon=False)
        ax.add_artist(ab)