# YuGiOH!-Elo Dashboard
## Using
### with uv
* uv run streamlit run main.py

### with pip
* pip install -r requirements.txt
* streamlit run main.py

# Databases 
## SQLite3 elo_database.db
### deck_statisitcs
* DeckID [Foreign Key, Primary Key] 
* Deck 
* Player
* Elo
* DGP
* Tier
* Matches
* Siege
* Remis
* Niederlagen
* Type
* Attack
* Combo
* Consistensy
* Control
* Recovery
* Resilience

### deck_history
* ID [Primary Key]
* DeckID [Foreign Key]
* DGP
* Elo
* Spiele
* Datum

### tournament
* ID [Primary Key]
* DeckID [Foreign Key]
* Datum 
* Win
* Draw
* Loss
* Type
* Result

## Sammlung.db
### deck_list
* ID [Primary Key]
* DeckID [Foreign Key]
* CollectionID [Foreign Key]
* Anzahl
* CardID

### collection
* ID [Primary Key, Foreign Key]
* Name
* Anzahl
* Set
* Preis

## Exerne Datenbanken
### Yugioh pro Datenbak (api v7)
* CardID [Primary Key]
* Price
* Image
* FrameType
* Name