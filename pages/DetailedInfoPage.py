import numpy as np
import pandas as pd
import streamlit as st
import VisualizationTools as vit

class DetailedInfoPage:
    '''
    Class for a page with a filter to choose a deck an see all
    statistics of this deck as well as its historics results
    '''

    def __init__(self):
        '''
        Init-Method
        :param df: dataframe with elo data
        :param hist_cols: column names with old elo data
        '''
        # build layout
        self.vis = vit.Visualization()
        self.df = st.session_state['deck_data']
        self.hist_cols = st.session_state['history_columns']
        self.__build_page_layout()

    def __build_page_layout(self):
        '''
        Method for creating the layout of the detailed info sub page.        
        '''
        deck, tournament = self.__header_row()
        # if deck was choosen display the stats
        if deck is None:
            st.stop()

        header_col = st.columns([2, 4])
        # Layout of deck KPI's, organisation in two rows
        # first row with deck name, tier, Icon, type, win/lose plot and stats spider
        # display deck name, icon and type
        head0 = header_col[0].container(border=False).columns(2)
        vit.load_and_display_image(f'./Deck_Icons/{deck["Deck"]}.png', 
                                   f'{deck["Tier"]},    {deck["Type"]}',
                                   pos=head0[0])
        if not deck['active']:
            head0[0].error('Deck ist pausiert')
        # win rate plot 
        vit.winrate_widget(head0[1], deck)

        # stats part
        elos = [deck[h] for h in self.hist_cols]
        elos.append(deck['Elo'])
        center_col = header_col[1].container(border=True)
        self.__metric_block(center_col, deck, elos, tournament)
        
        # disply stats spider plot
        categories = ['Attack', 'Control', 'Recovery','Consistensy', 'Combo', 'Resilience']
        fig = self.vis.make_spider_plot([int(deck[c]) for c in categories])
        head0[0].plotly_chart(fig, theme="streamlit", use_container_width=True)

        # Win stats
        local_wins, local_tops = 0, 0
        if not tournament.empty:
            local_wins = sum((tournament['Mode']=='Local')&(tournament['Standing']=='Win'))
            local_tops = sum((tournament['Mode']=='Local')&(tournament['Standing']=='Top'))
        self.__win_table('Wanderpokal', 'trophy', int(deck['Meisterschaft']['Win']['Wanderpokal']), head0[1])
        self.__win_table('Fun Pokal  ', 'star', int(deck['Meisterschaft']['Win']['Fun Pokal']), head0[1])
        self.__win_table('Local Win  ', 'medal', local_wins, head0[1])
        self.__win_table('Local Top  ', 'star', local_tops, head0[1])
        
        plot = header_col[1].radio('PlotAuswahl', ['Eloverlauf', 'Turniere'], horizontal=True, label_visibility='collapsed')
        if plot == 'Eloverlauf':
            fig = self.__line_plot(deck, elos)
        else:
            tournament.sort_values('Date', inplace=True, ignore_index=True)
            fig = self.vis.tournament_date_bar_plot(tournament)
        if fig is not None:
            header_col[1].plotly_chart(fig, theme=None, use_container_width=True)
        else:
            header_col[1].error('Es liegen keine Turnierergebnisse vor.')
    
    def __win_table(self, feature_name:str, symbol:str, amount:int, pos:str):
        localtop = ''.join(f':{symbol}: ' for _ in range(amount))
        pos.markdown(f"{feature_name:<15} "+localtop)
        
    def __header_row(self,):
        """
        """
        st.title(':trophy: Deck Details :trophy:', anchor='anchor_tag')
        # set data format and display a select box for choosing the deck of interest
        deck_ops = self.df['Deck'].astype(str).to_numpy()
        #deck_ops = np.append(np.array(['']), deck_ops)
        deck = st.selectbox('DeckChoosing', options=deck_ops, label_visibility='collapsed',
                            placeholder='Wähle ein deck')
        if len(deck)==0:
            return None, None
        idx_deck = int(self.df[self.df['Deck'] == deck].index.to_numpy())
        # tournament data
        tournament = st.session_state['tournament_data'].copy()
        tournament = tournament[tournament['Deck']==deck].reset_index(drop=True)
        # deck feature
        deck = self.df.loc[idx_deck]
        deck['percentage'] = ((len(self.df)-deck['EloPlatz']+1)/len(self.df)*100).astype(int)
        
        return deck, tournament
        
    def __metric_block(self, pos:st, deck, elos, tournament):
        """
        """
        dy = [int(deck[h]) for h in self.hist_cols]
        dy.append(deck['Elo'])
        dy = np.array(dy)
        dy = dy[dy>0]
        fig = self.vis.plot_metric(label="Aktuelle Elo", 
                                value=deck['Elo'],
                                x_data=self.hist_cols+['Elo'],
                                y_data=(dy-.95*min(dy)),
                                show_graph=True,
                                color_graph='rgba(0, 104, 201, 0.2)')
        pos.plotly_chart(fig, use_container_width=True, theme="streamlit")
        pos.markdown('---')
        inside = pos.columns([1, 1, 1, 1])
        # Metric Block
        inside[0].metric("Aktueller Rang:", value=f"{int(deck['EloPlatz'])}")
        inside[0].metric("Besser als:", f"{deck['percentage'] :.2f}%")
        
        inside[1].metric('Matches', int(deck['Matches']))
        inside[1].metric('Spiele:', f"{int(deck['Siege']+deck['Remis']+deck['Niederlage'])}")
        
        inside[2].metric("Beste Elo:", value=f"{int(max(elos))}")
        inside[2].metric("Letzte Änderung:", value=deck['Letzte 3 Monate'])

        inside[3].metric('Turniere', len(tournament))
        win, draw, loss  = 0, 0, 0
        if not tournament.empty:
            win = tournament['Win'].sum()
            draw = tournament['Draw'].sum()
            loss = tournament['Loss'].sum()
        inside[3].metric('Turnier Ergebnisse S/U/N', f"{win}/{draw}/{loss}")

    def __line_plot(self, deck, elos):
        '''
        Method for creating a lineplot of the elo points of the last year
        :param df: dataframe with historic elo
        :param hist_cols: list wit historic columns
        :param idx: index of the current deck
        :return fig: plotly figure object
        '''
        # generate plot dataframe with elo, date and mean of the last year
        df_plot = pd.DataFrame(columns=['Elo'])
        df_plot['Elo'] = elos[-6:]
        df_plot['Datum'] = self.hist_cols[-5:]+['Elo']
        df_plot = df_plot[df_plot['Elo'] > 0].reset_index(drop=True)
        df_plot['Mittelw.'] = df_plot['Elo'].mean()
        # generate plotly figure and plot as well as layout of the figure
        return self.vis.lineplot(df_plot["Datum"], df_plot["Elo"], df_plot["Mittelw."])