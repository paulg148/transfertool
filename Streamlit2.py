import streamlit as st
import pandas as pd
import numpy as np


file_path = "spieler_statistik.csv"
df = pd.read_csv(file_path)

def get_top_eleven_by_minutes(team_name):
    team_data = df[df["team"] == team_name]
    
    sorted_team_data = team_data.sort_values(by="total_minutes", ascending=False)
    top_players = sorted_team_data.head(11)

    position_order = ["GK", "LB", "CB", "RB", "WB", "LM", "LW", "DM", "MF", "CM", "RM", "RW", "AM", "FW"]
    
    position_priority = {}
    for i in range(len(position_order)):
        position_priority[position_order[i]] = i

    ###{"GK": 0, "LB": 1, "CB": 2, ...}###
    
    top_players["position_priority"] = top_players["most_frequent_position"].map(position_priority) ###Jedem Spieler wird deine Nummer an Priorität zugewiesen
    top_players = top_players.sort_values(by=["position_priority", "total_minutes"], ascending=[True, False])
    top_players = top_players.drop(columns="position_priority")
    
    return top_players
    



def get_top_20_attributes_for_player(player_name, df, position_priorities):
    player_row = df[df['player'] == player_name] 

    player_position = player_row.iloc[0]['most_frequent_position']

    relevant_attributes = position_priorities.get(player_position)

    player_attributes = player_row[relevant_attributes].iloc[0]
    top_20_attributes = player_attributes.sort_values(ascending=False).head(20)
    
    return top_20_attributes



def get_top_20_attributes_for_similar_players(similar_players, top_20_attributes, df):
    similar_players_data = {}

    for player_name in similar_players:
        player_row = df[df['player'] == player_name]
        player_attributes = player_row[top_20_attributes.index].iloc[0]
        similar_players_data[player_name] = player_attributes

    return pd.DataFrame(similar_players_data).T


#####TEIL2######





#####APP-AUFBAU########

st.title("Startelf-Visualisierung")

teams = df["team"].unique()
selected_team = st.selectbox("Wähle ein Team aus", options=sorted(teams))


starting_eleven = get_top_eleven_by_minutes(selected_team)

st.sidebar.subheader("Spielerfenster")
st.subheader(f"Startelf für {selected_team}")

player_names = starting_eleven["player"].tolist()
selected_player = st.selectbox("Wähle einen Spieler aus, um ihn zu entfernen", options=[""] + player_names)

updated_starting_eleven = starting_eleven.copy()

if selected_player:
    updated_starting_eleven.loc[updated_starting_eleven["player"] == selected_player, "player"] = "____________________"

for _, player in updated_starting_eleven.iterrows():
    st.write(f"**{player['most_frequent_position']}** - {player['player']}")


######ZWEITER TEIL#######


from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cleaned_file_path = "cleaned_final_stats_with_position90.csv"
df = pd.read_csv(cleaned_file_path)

X = df.drop(columns=['most_frequent_position', 'player', 'team', 'ages', 'nation', 'pos', 'marktwert']) 
player_names = df['player'].values  # Namen der Spieler

position_priorities = {
    'CB': ['Progression_PrgC', 'Progression_PrgP', 'Pass_Types_Live', 'Total_Cmp', 'Short_Cmp', 'Medium_Cmp', 'Long_Cmp', 'Tackles_Def_3rd', 'Tackles_Mid_3rd', 'Challenges_Tkl', 'Challenges_Tkl%', 'Blocks_Blocks', 'Blocks_Sh', 'Blocks_Pass', 'Touches_Def_Pen', 'Touches_Def_3rd', 'Touches_Mid_3rd' ,'Take-Ons_Succ%', 'Carries_PrgDist', 'Aerial_Duels_Won', 'Aerial_Duels_Lost', 'Aerial_Duels_Won%'],
    'WB': ["Tackles_Def_3rd", "Tackles_Mid_3rd", "Tackles_Att_3rd", "Challenges_Tkl", "Tkl+Int", "Aerial_Duels_Won%", "Expected_xA", "KP", "PPA", "CrsPA", "Pass_Types_Live", "Pass_Types_TB", "Pass_Types_CK","Touches_Touches",	"Touches_Mid_3rd", "Touches_Att_3rd", "Take-Ons_Succ%", "Carries_PrgDist", "Expected_xAG", "Progression_PrgC", "Progression_PrgP", "Progression_PrgR"],  
    'MF': ["Tackles_Def_3rd", "Tackles_Mid_3rd", "Tackles_Att_3rd",	"Challenges_Tkl", "Tkl+Int", "SCA_Types_PassLive", "SCA_Types_TO", "Total_Cmp", "Total_TotDist", "Total_PrgDist", "Short_Cmp", "Medium_Cmp", "Long_Cmp", "KP","PPA", "CrsPA", "Pass_Types_Live", "Pass_Types_TB", "Pass_Types_Sw", "Touches_Mid_3rd", "Touches_Att_3rd", "Carries_Carries", "Standard_Sh", "Progression_PrgC", "Progression_PrgP", "Progression_PrgR"], 
    'WM': ["Tackles_Att_3rd", "Challenges_Tkl%", "Blocks_Pass", "SCA_SCA", "Aerial_Duels_Won%", "Total_Cmp", "Short_Cmp", "Expected_xA", "KP", "PPA", "CrsPA", "Pass_Types_TB", "Pass_Types_Crs", "Touches_Att_3rd", "Touches_Att_Pen", "Take-Ons_Att", "Carries_PrgDist", "Standard_Sh", "Progression_PrgC", "Progression_PrgR"],
    'FW': ["Tackles_Att_3rd", "SCA_Types_PassLive", "SCA_Types_TO", "Aerial_Duels_Won", "Short_Cmp", "Medium_Cmp", "KP", "PPA", "CrsPA", "Pass_Types_TB", "Pass_Types_Crs", "Touches_Att_3rd", "Touches_Att_Pen", "Take-Ons_Att", "Carries_PrgDist", "Standard_Sh", "Expected_npxG", "Expected_npxG+xAG", "Progression_PrgC", "Progression_PrgP", "Progression_PrgR"]
    }

def adjust_features_for_position(df, position, X):
    position_attributes = position_priorities.get(position)
    
    weights = np.ones(X.shape[1])   ###Eine Tabelle mit 1en im selben Format wie X
    
    for i, column in enumerate(X.columns):
        if column in position_attributes:
            weights[i] = 3     ###Wenn es ein position attribut ist durch 10 ersetzen

    weighted_X = X * weights    ###Ausprägungen mit den Faktoren multiplizieren
    return weighted_X


player_name = selected_player

player_idx = None
for i, name in enumerate(player_names):
    if name == player_name:
        player_idx = i
        break

if player_idx is None:
    st.sidebar.error("Wähle einen Spieler aus!")
else:
    player_info = df.loc[player_idx]
    st.sidebar.markdown(f"""
        **👤 Name:** {player_info['player']}  
        **🎂 Alter:** {player_info['ages']}  
        **🌍 Nation:** {player_info['nation']}  
        **🏟️ Verein:** {player_info['team']}          
        **💰 Marktwert:** {player_info['marktwert']}
    """)
    

    min_age, max_age = st.sidebar.slider(
        "Alter auswählen", 
        min_value=int(df['ages'].min()), 
        max_value=int(df['ages'].max()), 
        value=(int(df['ages'].min()), int(df['ages'].max()))
    )

    min_value, max_value = st.sidebar.slider(
        "Marktwert auswählen (in Millionen)",
        min_value=float(df['marktwert'].min()), 
        max_value=float(df['marktwert'].max()),
        value=(float(df['marktwert'].min()), float(df['marktwert'].max())))

    filtered_df = df[(df['ages'] >= min_age) & (df['ages'] <= max_age) & (df['marktwert'] >= min_value) & (df['marktwert'] <= max_value)]

    if player_name not in filtered_df['player'].values:
        player_row = df[df['player'] == player_name]
        filtered_df = pd.concat([filtered_df, player_row])

    filtered_X = filtered_df.drop(columns=['most_frequent_position', 'player', 'team', 'ages', 'nation', 'pos', 'marktwert'])

    filtered_player_idx = filtered_df.index[filtered_df['player'] == player_name].tolist()

    filtered_player_idx = filtered_player_idx[0]

    player_position = df.loc[player_idx]['most_frequent_position']
    X_adjusted = adjust_features_for_position(filtered_df, player_position, filtered_X)

    pca = PCA(n_components=20)
    knn = NearestNeighbors(n_neighbors=6)

    X_pca_adjusted = pca.fit_transform(X_adjusted)
    knn.fit(X_pca_adjusted)

    decimals, indices = knn.kneighbors([X_pca_adjusted[filtered_df.index.get_loc(filtered_player_idx)]])  ###Erstellt eine Liste an Indices, die in der "filtered_df" die Nearest Neighbors zeigen 
    similar_players_idx = indices[0][1:] 

    st.sidebar.subheader(f"Ähnlichste Spieler zu {player_name}")
    for idx in similar_players_idx:
        similar_player_info = filtered_df.iloc[idx]
        st.sidebar.markdown(f"""
        **Name:** {similar_player_info['player']}  
        **Alter:** {similar_player_info['ages']}  
        **Nation:** {similar_player_info['nation']}  
        **Verein:** {similar_player_info['team']}            
        **Marktwert:** {similar_player_info['marktwert']}  
        """)

    player_name = selected_player 
    top_20_attributes = get_top_20_attributes_for_player(player_name, df, position_priorities)

    similar_players = filtered_df.iloc[similar_players_idx]['player'].tolist()

    all_players_data = get_top_20_attributes_for_similar_players([player_name] + similar_players, top_20_attributes, df)

    attribute_translation = {
        'SCA_Types_PassLive': 'SCA rollend',
        'SCA_Types_PassDead': 'SCA ruhend',
        'SCA_Types_TO': 'SCA Zweikampf',
        'SCA_Types_Fld': 'SCA Gefoult',
        'Aerial_Duels_Won': 'Kopfbälle - Gewonnen',
        'Aerial_Duels_Lost': 'Kopfbälle - Verloren',
        'Aerial_Duels_Won%': 'Kopfbälle - Quote',
        'Total_Cmp': 'Erfolgreiche Pässe',
        'Total_Cmp%': 'Passquote',
        'Total_TotDist': 'Pässe - GesEntf',
        'Total_PrgDist': 'Pässe - ProgEntf',
        'Short_Cmp': 'Kurze Pässe - Erfolgreich',
        'Short_Cmp%': 'Kurze Pässe - Quote',
        'Medium_Cmp': 'Mittel Pässe - Erfolgreich',
        'Medium_Cmp%': 'Mittel Pässe - Quote',
        'Long_Cmp': 'Lange Pässe - Erfolgreich',
        'Long_Cmp%': 'Lange Pässe - Quote',
        'Expected_xA': 'xA',
        'Expected_A-xAG': 'A-xAG',
        'KP': 'xWichtige Pässe',
        'PPA': 'xPässe in 16er',
        'CrsPA': 'xFlanken in 16er',
        'Pass_Types_Live': 'Passarten - rollend',
        'Pass_Types_Dead': 'Passarten - ruhend',
        'Pass_Types_FK': 'Passarten - Freistoß',
        'Pass_Types_TB': 'Passarten - Steilpass',
        'Pass_Types_Sw': 'Passarten - Wichtiger Pass',
        'Pass_Types_Crs': 'Passarten - Flanke',
        'Pass_Types_TI': 'Passarten - Einwurf',
        'Pass_Types_CK': 'Passarten - Eckball',
        'Outcomes_Blocks': 'Gespielte Pässe geblockt',
        'Touches_Touches': 'Ballaktionen',
        'Touches_Def_Pen': 'Ballaktionen - Def, Strafr',
        'Touches_Def_3rd': 'Ballaktionen - Def, 1/3',
        'Touches_Mid_3rd': 'Ballaktionen - Mit, 1/3',
        'Touches_Att_3rd': 'Ballaktionen - Off, 1/3',
        'Touches_Att_Pen': 'Ballaktionen - Off, Strafr',
        'Take-Ons_Att': 'Offensivzweikampf - Succ',
        'Take-Ons_Succ%': 'Offensivzweikampf - Quote',
        'Carries_Carries': 'Ballführung - Anzahl',
        'Carries_TotDist': 'Ballführung - GesEntf',
        'Carries_PrgDist': 'Ballführung - ProgEntf',
        'Carries_Mis': 'Ballführung - Fehlkontrolle',
        'Standard_Gls': 'Tore',
        'Standard_Sh': 'Schüsse',
        'Standard_Dist': 'Schussdistanz',
        'Standard_FK': 'Freistöße',
        'Expected_npxG': 'non-Penalty xG',
        'Performance_Ast': 'Vorlagen',
        'Expected_xAG': 'xAG',
        'Expected_npxG+xAG': 'non-Penalty xG+xAG',
        'Progression_PrgC': 'Progrerssive Dribblings',
        'Progression_PrgP': 'Prgressive Pässe',
        'Progression_PrgR': 'Progressive Ballannahmen',
        'Tackles_Tkl': 'Zweikämpfe',
        'Tackles_TklW': 'Zweikämpfe gewonnen',
        'Tackles_Def_3rd': 'Zweikämpfe - Def, 1/3',
        'Tackles_Mid_3rd': 'Zweikämpfe - Mit, 1/3',
        'Tackles_Att_3rd': 'Zweikämpfe - Off, 1/3',
        'Challenges_Tkl': 'Def. Dribbling',
        'Challenges_Tkl%': 'Def. Dribbling gewonnen',
        'Blocks_Blocks': 'Def Blocks',
        'Blocks_Sh': 'Schüsse geblockt',
        'Blocks_Pass': 'Pässe geblockt',
        'Tkl+Int': 'Tkl + Blocks',
        'Err': 'Fehler zu Schüssen',
        'SCA_SCA': 'SCA'
    }

    def translate_attributes(df, translation_dict):
        df = df.rename(columns=translation_dict)
        return df

    df_translated = translate_attributes(all_players_data, attribute_translation)

    st.subheader("Spielervergleich anhand der wichtigsten Positionsattribute")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_translated, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.ylabel("Spieler")
    plt.xlabel("Attribute")
    plt.yticks(rotation=0)
    st.pyplot(plt)




