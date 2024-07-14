import streamlit as st
from streamlit_timeline import timeline

def timeline_for_decks():
    '''
    '''
    time_json = {'title':{"text": {
                        "headline": "Top 5",
                        "text": "Die besten Decks im Vergleich"
                        }},
                'events':[]}
    
    for date in st.session_state['history_columns']:
        # setup date
        datum = {"year":str(date[3:]),
                "month":str(int(date[:2]))}
        # set up placements 
        test = st.session_state['deck_data'][[date, 'Deck']].copy().sort_values(by=date, ascending=False).reset_index(drop=True)
        text = {'text': f"<p>{test.at[0, 'Deck']} {int(test.at[0, date])}<p>\
                        <p>{test.at[1, 'Deck']} {int(test.at[1, date])}<p>\
                        <p>{test.at[2, 'Deck']} {int(test.at[2, date])}<p>\
                        <p>{test.at[3, 'Deck']} {int(test.at[3, date])}<p>\
                        <p>{test.at[4, 'Deck']} {int(test.at[4, date])}<p>",
                'headline':""}
        event = {"start_date": datum,
                "text":text}
        time_json['events'].append(event)
    timeline(time_json, height=410)