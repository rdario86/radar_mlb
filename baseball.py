import streamlit as st
import pandas as pd
import numpy as np
import statsapi
import time
import datetime
import calendar
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import poisson
from scipy.stats import binom

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Predicción MLB Automatizada", layout="wide", page_icon="⚾")

st.title("⚾ Predicción MLB: Radar Diario Automatizado")
st.markdown("Proyección Sabermétrica")
st.markdown("---")

# --- PARÁMETROS SABERMÉTRICOS FIJOS ---
MAX_DEPTH_ELO = 5       
PESO_RACHA = 0.08       
PESO_H2H = 0.12         
PESO_PITAGORICO = 0.10  
PESO_SPLITS = 0.10      
LINEA_TOTALES = 8.5     

MLB_TEAM_WHITELIST = [
    "Arizona Diamondbacks", "Atlanta Braves", "Baltimore Orioles", "Boston Red Sox", 
    "Chicago Cubs", "Chicago White Sox", "Cincinnati Reds", "Cleveland Guardians", 
    "Colorado Rockies", "Detroit Tigers", "Houston Astros", "Kansas City Royals", 
    "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins", "Milwaukee Brewers", 
    "Minnesota Twins", "New York Mets", "New York Yankees", "Athletics", 
    "Philadelphia Phillies", "Pittsburgh Pirates", "San Diego Padres", "San Francisco Giants", 
    "Seattle Mariners", "St. Louis Cardinals", "Tampa Bay Rays", "Texas Rangers", 
    "Toronto Blue Jays", "Washington Nationals"
]

# --- FUNCIONES DE APOYO ---
def get_recent_form(team, df, n=10):
    team_games = df[(df['Local'] == team) | (df['Visitante'] == team)].tail(n)
    if len(team_games) == 0: return 0.5
    wins = sum(1 for _, row in team_games.iterrows() 
               if (row['Local'] == team and row['Carreras_Local'] > row['Carreras_Visitante']) or 
                  (row['Visitante'] == team and row['Carreras_Visitante'] > row['Carreras_Local']))
    return wins / len(team_games)

def get_h2h_wins(team_a, team_b, df, n=5):
    h2h = df[((df['Local'] == team_a) & (df['Visitante'] == team_b)) | 
             ((df['Local'] == team_b) & (df['Visitante'] == team_a))]
    last_5 = h2h.tail(n)
    if len(last_5) == 0: return 0.5
    wins = sum(1 for _, row in last_5.iterrows() 
               if (row['Local'] == team_a and row['Carreras_Local'] > row['Carreras_Visitante']) or 
                  (row['Visitante'] == team_a and row['Carreras_Visitante'] > row['Carreras_Local']))
    return wins / len(last_5)

def get_run_metrics(team, df, n=10):
    team_games = df[(df['Local'] == team) | (df['Visitante'] == team)].tail(n)
    if len(team_games) == 0: return 4.5, 4.5
    runs_scored = sum(row['Carreras_Local'] if row['Local'] == team else row['Carreras_Visitante'] for _, row in team_games.iterrows())
    runs_allowed = sum(row['Carreras_Visitante'] if row['Local'] == team else row['Carreras_Local'] for _, row in team_games.iterrows())
    return runs_scored / len(team_games), runs_allowed / len(team_games)

def get_team_record(team, df):
    wins = sum((df['Local'] == team) & (df['Carreras_Local'] > df['Carreras_Visitante'])) + \
           sum((df['Visitante'] == team) & (df['Carreras_Visitante'] > df['Carreras_Local']))
    losses = sum((df['Local'] == team) & (df['Carreras_Local'] < df['Carreras_Visitante'])) + \
             sum((df['Visitante'] == team) & (df['Carreras_Visitante'] < df['Carreras_Local']))
    return f"{wins}-{losses}"

def get_pythagorean_luck(team, df):
    team_games = df[(df['Local'] == team) | (df['Visitante'] == team)]
    if len(team_games) == 0: return 0.0
    rs = sum(row['Carreras_Local'] if row['Local'] == team else row['Carreras_Visitante'] for _, row in team_games.iterrows())
    ra = sum(row['Carreras_Visitante'] if row['Local'] == team else row['Carreras_Local'] for _, row in team_games.iterrows())
    if rs + ra == 0: return 0.0
    exp = 1.83
    pyth_exp = (rs**exp) / ((rs**exp) + (ra**exp)) if (rs**exp + ra**exp) > 0 else 0.5
    wins = sum(1 for _, row in team_games.iterrows() 
               if (row['Local'] == team and row['Carreras_Local'] > row['Carreras_Visitante']) or 
                  (row['Visitante'] == team and row['Carreras_Visitante'] > row['Carreras_Local']))
    actual_win_pct = wins / len(team_games)
    return pyth_exp - actual_win_pct

def get_splits_win_pct(home_team, away_team, df):
    home_games = df[df['Local'] == home_team]
    home_win_pct = sum(1 for _, row in home_games.iterrows() if row['Carreras_Local'] > row['Carreras_Visitante']) / len(home_games) if len(home_games) > 0 else 0.5
    away_games = df[df['Visitante'] == away_team]
    away_win_pct = sum(1 for _, row in away_games.iterrows() if row['Carreras_Visitante'] > row['Carreras_Local']) / len(away_games) if len(away_games) > 0 else 0.5
    return home_win_pct, away_win_pct

def get_hybrid_run_projection(away_team, home_team, df):
    rs_a_10, ra_a_10 = get_run_metrics(away_team, df, 10)
    rs_h_10, ra_h_10 = get_run_metrics(home_team, df, 10)
    base_runs_away = (rs_a_10 + ra_h_10) / 2.0
    base_runs_home = (rs_h_10 + ra_a_10) / 2.0

    away_all = df[(df['Local'] == away_team) | (df['Visitante'] == away_team)]
    if len(away_all) > 0:
        away_rs_g = sum(row['Carreras_Visitante'] if row['Visitante'] == away_team else row['Carreras_Local'] for _, row in away_all.iterrows()) / len(away_all)
    else: away_rs_g = 1.0
        
    home_all = df[(df['Local'] == home_team) | (df['Visitante'] == home_team)]
    if len(home_all) > 0:
        home_ra_g = sum(row['Carreras_Visitante'] if row['Local'] == home_team else row['Carreras_Local'] for _, row in home_all.iterrows()) / len(home_all)
        home_rs_g = sum(row['Carreras_Local'] if row['Local'] == home_team else row['Carreras_Visitante'] for _, row in home_all.iterrows()) / len(home_all)
    else: home_ra_g, home_rs_g = 1.0, 1.0

    away_split = df[df['Visitante'] == away_team]
    away_rs_s = away_split['Carreras_Visitante'].mean() if len(away_split) > 0 else away_rs_g
    
    home_split = df[df['Local'] == home_team]
    home_ra_s = home_split['Carreras_Visitante'].mean() if len(home_split) > 0 else home_ra_g
    home_rs_s = home_split['Carreras_Local'].mean() if len(home_split) > 0 else home_rs_g

    m_off_away = away_rs_s / away_rs_g if away_rs_g > 0 else 1.0
    m_def_home = home_ra_s / home_ra_g if home_ra_g > 0 else 1.0
    m_off_home = home_rs_s / home_rs_g if home_rs_g > 0 else 1.0
    
    proj_away = base_runs_away * m_off_away * m_def_home
    proj_home = base_runs_home * m_off_home * 1.0
    return round(proj_away, 2), round(proj_home, 2)

def get_starting_pitchers(juego):
    hp = juego.get('home_probable_pitcher', '')
    ap = juego.get('away_probable_pitcher', '')
    
    if (not hp or hp == 'TBD' or not ap or ap == 'TBD'):
        try:
            game_id = juego.get('game_id')
            if game_id:
                box = statsapi.boxscore_data(game_id)
                if not hp or hp == 'TBD':
                    hp_list = box.get('home', {}).get('pitchers', [])
                    if hp_list:
                        hp = box.get('playerInfo', {}).get(f"ID{hp_list[0]}", {}).get('fullName', 'TBD')
                if not ap or ap == 'TBD':
                    ap_list = box.get('away', {}).get('pitchers', [])
                    if ap_list:
                        ap = box.get('playerInfo', {}).get(f"ID{ap_list[0]}", {}).get('fullName', 'TBD')
        except: pass
            
    return hp, ap

def get_pitcher_whip(pitcher_name, fecha_corte):
    avg_whip = 1.30 
    if not pitcher_name or pitcher_name == 'TBD': return avg_whip
    try:
        players = statsapi.lookup_player(pitcher_name)
        if not players: return avg_whip
        player_id = players[0]['id']
        try:
            raw_data = statsapi.get('people', {'personIds': player_id, 'hydrate': 'stats(group=[pitching],type=[gameLog])'})
            if 'people' in raw_data and len(raw_data['people']) > 0:
                person = raw_data['people'][0]
                if 'stats' in person:
                    for stat_block in person['stats']:
                        if stat_block.get('type', {}).get('displayName') == 'gameLog':
                            splits = stat_block.get('splits', [])
                            if splits:
                                valid_splits = [s for s in splits if s.get('date', '') < fecha_corte]
                                valid_splits.sort(key=lambda x: x.get('date', ''), reverse=True)
                                last_7 = valid_splits[:7]
                                
                                total_hits = 0; total_bb = 0; total_outs = 0
                                for game in last_7:
                                    g_stats = game.get('stat', {})
                                    total_hits += int(g_stats.get('hits', 0))
                                    total_bb += int(g_stats.get('baseOnBalls', 0))
                                    ip_str = str(g_stats.get('inningsPitched', '0.0'))
                                    if '.' in ip_str:
                                        full, frac = ip_str.split('.')
                                        total_outs += (int(full) * 3) + int(frac)
                                    else:
                                        total_outs += int(ip_str) * 3
                                        
                                if total_outs > 0: return round((total_hits + total_bb) / (total_outs / 3.0), 2)
                                else: return avg_whip
        except Exception: pass
        return avg_whip
    except: return avg_whip

def get_pitcher_hand(pitcher_name):
    """Devuelve 'R', 'L' o 'U' según la mano de lanzar del pitcher."""
    if not pitcher_name or pitcher_name == 'TBD':
        return 'U'
    try:
        players = statsapi.lookup_player(pitcher_name)
        if not players:
            return 'U'
        pid = players[0]['id']
        person = statsapi.get('person', {'personId': pid})
        hand = person.get('people', [{}])[0].get('pitchHand', {}).get('code', 'U')
        return hand
    except:
        return 'U'

def get_hr_hunters(anio, fecha_hoy):
    try:
        juegos_hoy = statsapi.schedule(date=fecha_hoy, sportId=1)
        equipos_hoy = {}
        team_game_map = {}
        for juego in juegos_hoy:
            if juego.get('status', '') not in ['Postponed', 'Cancelled']:
                home_id = juego.get('home_id')
                away_id = juego.get('away_id')
                equipos_hoy[home_id] = {'condicion': 'Local', 'status': juego.get('status')}
                equipos_hoy[away_id] = {'condicion': 'Visitante', 'status': juego.get('status')}
                team_game_map[home_id] = juego
                team_game_map[away_id] = juego

        data = statsapi.get('stats_leaders', {'leaderCategories': 'homeRuns', 'season': anio, 'limit': 80, 'statGroup': 'hitting'})
        if not data or 'leagueLeaders' not in data or len(data['leagueLeaders']) == 0:
            return []

        leaders = data['leagueLeaders'][0].get('leaders', [])
        jugadores_activos = []
        for p in leaders:
            team_id = p.get('team', {}).get('id')
            if team_id in equipos_hoy:
                p['condicion_hoy'] = equipos_hoy[team_id]['condicion']
                p['team_name'] = p.get('team', {}).get('name', 'Unknown')
                p['game_status'] = equipos_hoy[team_id]['status']
                p['team_id'] = team_id
                jugadores_activos.append(p)

        resultados = []
        ayer_dt = datetime.datetime.strptime(fecha_hoy, '%Y-%m-%d') - datetime.timedelta(days=1)
        fecha_ayer_str = ayer_dt.strftime('%Y-%m-%d')

        # Cache de manos de pitchers para evitar repetir llamadas
        pitcher_hand_cache = {}

        for p in jugadores_activos:
            p_id = p.get('person', {}).get('id')
            p_name = p.get('person', {}).get('fullName')
            team_name = p.get('team_name', 'Unknown')
            condicion = p.get('condicion_hoy', 'Visitante')
            team_id = p.get('team_id')
            game_status = p.get('game_status', '')

            # Obtener juego y pitcher rival
            juego = team_game_map.get(team_id)
            opposing_pitcher = None
            if juego:
                if condicion == 'Local':
                    opposing_pitcher = juego.get('away_probable_pitcher', '')
                else:
                    opposing_pitcher = juego.get('home_probable_pitcher', '')
                if not opposing_pitcher or opposing_pitcher == 'TBD':
                    hp, ap = get_starting_pitchers(juego)
                    opposing_pitcher = ap if condicion == 'Local' else hp

            raw_data = statsapi.get('people', {'personIds': p_id, 'hydrate': 'stats(group=[hitting],type=[season,gameLog])'})
            person = raw_data.get('people', [{}])[0]
            stats_blocks = person.get('stats', [])

            season_ab = 1; season_hr = 0
            l10_hr = 0; l10_ab = 0
            hr_hoy_real = 0; ab_hoy_real = 0
            dio_jonron_ayer = False

            for block in stats_blocks:
                if block.get('type', {}).get('displayName') == 'season':
                    season_ab = int(block.get('splits', [{}])[0].get('stat', {}).get('atBats', 1))
                    season_hr = int(block.get('splits', [{}])[0].get('stat', {}).get('homeRuns', 0))
                elif block.get('type', {}).get('displayName') == 'gameLog':
                    splits = block.get('splits', [])
                    valid_splits = [s for s in splits if s.get('date', '') < fecha_hoy]
                    valid_splits.sort(key=lambda x: x.get('date', ''), reverse=True)

                    for game in valid_splits[:10]:
                        g_stats = game.get('stat', {})
                        if game.get('date', '') == fecha_ayer_str and int(g_stats.get('homeRuns', 0)) > 0:
                            dio_jonron_ayer = True
                        l10_hr += int(g_stats.get('homeRuns', 0))
                        l10_ab += int(g_stats.get('atBats', 0))

                    for game in splits:
                        if game.get('date') == fecha_hoy:
                            hr_hoy_real += int(game.get('stat', {}).get('homeRuns', 0))
                            ab_hoy_real += int(game.get('stat', {}).get('atBats', 0))

            if dio_jonron_ayer: continue
            if game_status in ['Final', 'Game Over'] and ab_hoy_real == 0: continue

            season_hr = max(0, season_hr - hr_hoy_real)
            season_ab = max(1, season_ab - ab_hoy_real)
            l10_ab = max(1, l10_ab)

            # Índice base (sin platoon)
            hr_index = ((season_hr / season_ab * 0.3) + (l10_hr / l10_ab * 0.7)) * (1.10 if condicion == 'Local' else 1.0)

            # Ajuste por mano del lanzador rival (platoon)
            factor_platoon = 1.0
            if opposing_pitcher and opposing_pitcher != 'TBD':
                if opposing_pitcher not in pitcher_hand_cache:
                    pitcher_hand_cache[opposing_pitcher] = get_pitcher_hand(opposing_pitcher)
                pitcher_hand = pitcher_hand_cache[opposing_pitcher]
                bat_hand = person.get('batSide', {}).get('code', 'U')
                if pitcher_hand in ('R', 'L') and bat_hand in ('R', 'L'):
                    if pitcher_hand != bat_hand:
                        if bat_hand == 'L':
                            factor_platoon = 1.25
                        else:
                            factor_platoon = 1.15

            hr_index *= factor_platoon
            hr_index_rounded = round(hr_index, 4)

            prob_hr_game = 1 - (1 - hr_index) ** 4
            prob_hr_pct = int(round(prob_hr_game * 100))

            eval_str = "⏳ Pendiente"
            if game_status in ['Final', 'Game Over']:
                if ab_hoy_real == 0:
                    eval_str = "🚫 No jugó"
                else:
                    eval_str = "✅ Acierto" if hr_hoy_real > 0 else "❌ Fallo"

            resultados.append({
                "⚾ Bateador": p_name,
                "👕 Equipo": team_name,
                "🏟️ Condición": condicion,
                "🏆 HR Año": season_hr,
                "🔥 HR (L10)": l10_hr,
                "📊 Turnos (L10)": l10_ab,
                "📝 Evaluación": eval_str,
                "score": hr_index_rounded,
                "prob_hr_game": prob_hr_pct
            })

        resultados.sort(key=lambda x: (x['score'], x['⚾ Bateador']), reverse=True)
        resultados = resultados[:4]

        tabla_final = []
        for r in resultados:
            tabla_final.append({
                "⚾ Bateador": r["⚾ Bateador"],
                "👕 Equipo": r["👕 Equipo"],
                "🏟️ Condición": r["🏟️ Condición"],
                "🏆 HR Año": r["🏆 HR Año"],
                "🔥 HR (L10)": r["🔥 HR (L10)"],
                "📈 Ratio de Poder (L10)": f"{r['🔥 HR (L10)']} HR / {r['📊 Turnos (L10)']} VB",
                "🎯 Prob. HR (Partido)": f"{r['prob_hr_game']}%",
                "📝 Evaluación": r["📝 Evaluación"]
            })
        return tabla_final
    except Exception:
        return []

def get_hit_hunters(anio, fecha_hoy):
    """Devuelve los 4 bateadores con mayor probabilidad de dar al menos 1 hit hoy."""
    try:
        juegos_hoy = statsapi.schedule(date=fecha_hoy, sportId=1)
        equipos_hoy = {}
        for juego in juegos_hoy:
            if juego.get('status', '') not in ['Postponed', 'Cancelled']:
                equipos_hoy[juego.get('home_id')] = {'condicion': 'Local', 'status': juego.get('status')}
                equipos_hoy[juego.get('away_id')] = {'condicion': 'Visitante', 'status': juego.get('status')}

        data = statsapi.get('stats_leaders', {'leaderCategories': 'battingAverage', 'season': anio, 'limit': 80, 'statGroup': 'hitting'})
        if not data or 'leagueLeaders' not in data or len(data['leagueLeaders']) == 0:
            return []

        leaders = data['leagueLeaders'][0].get('leaders', [])
        jugadores_activos = []
        for p in leaders:
            team_id = p.get('team', {}).get('id')
            if team_id in equipos_hoy:
                p['condicion_hoy'] = equipos_hoy[team_id]['condicion']
                p['team_name'] = p.get('team', {}).get('name', 'Unknown')
                p['game_status'] = equipos_hoy[team_id]['status']
                jugadores_activos.append(p)

        resultados = []
        ayer_dt = datetime.datetime.strptime(fecha_hoy, '%Y-%m-%d') - datetime.timedelta(days=1)
        fecha_ayer_str = ayer_dt.strftime('%Y-%m-%d')

        for p in jugadores_activos:
            p_id = p.get('person', {}).get('id')
            p_name = p.get('person', {}).get('fullName')
            team_name = p.get('team_name', 'Unknown')
            condicion = p.get('condicion_hoy', 'Visitante')
            game_status = p.get('game_status', '')

            raw_data = statsapi.get('people', {'personIds': p_id, 'hydrate': 'stats(group=[hitting],type=[season,gameLog])'})
            person = raw_data.get('people', [{}])[0]
            stats_blocks = person.get('stats', [])

            season_ab = 1; season_hits = 0
            l10_hits = 0; l10_ab = 0
            hits_hoy_real = 0; ab_hoy_real = 0
            dio_2hits_ayer = False

            for block in stats_blocks:
                if block.get('type', {}).get('displayName') == 'season':
                    season_ab = int(block.get('splits', [{}])[0].get('stat', {}).get('atBats', 1))
                    season_hits = int(block.get('splits', [{}])[0].get('stat', {}).get('hits', 0))
                elif block.get('type', {}).get('displayName') == 'gameLog':
                    splits = block.get('splits', [])
                    valid_splits = [s for s in splits if s.get('date', '') < fecha_hoy]
                    valid_splits.sort(key=lambda x: x.get('date', ''), reverse=True)

                    for game in valid_splits[:10]:
                        g_stats = game.get('stat', {})
                        if game.get('date', '') == fecha_ayer_str and int(g_stats.get('hits', 0)) >= 2:
                            dio_2hits_ayer = True
                        l10_hits += int(g_stats.get('hits', 0))
                        l10_ab += int(g_stats.get('atBats', 0))

                    for game in splits:
                        if game.get('date') == fecha_hoy:
                            hits_hoy_real += int(game.get('stat', {}).get('hits', 0))
                            ab_hoy_real += int(game.get('stat', {}).get('atBats', 0))

            # Filtro anti‑racha: si ayer dio 2+ hits, lo descartamos
            if dio_2hits_ayer: continue

            # Si el partido terminó y no jugó, lo excluimos
            if game_status in ['Final', 'Game Over'] and ab_hoy_real == 0: continue

            season_hits = max(0, season_hits - hits_hoy_real)
            season_ab = max(1, season_ab - ab_hoy_real)
            l10_ab = max(1, l10_ab)

            # Promedio de bateo combinado
            avg_index = (season_hits / season_ab * 0.3) + (l10_hits / l10_ab * 0.7)

            # *** NUEVO: probabilidad de al menos 1 hit en 4 turnos ***
            prob_1hit = 1 - (1 - avg_index) ** 4
            prob_1hit_pct = int(round(prob_1hit * 100))

            eval_str = "⏳ Pendiente"
            if game_status in ['Final', 'Game Over']:
                if ab_hoy_real == 0:
                    eval_str = "🚫 No jugó"
                else:
                    # Acierto si dio al menos 1 hit
                    eval_str = "✅ Acierto" if hits_hoy_real >= 1 else "❌ Fallo"

            resultados.append({
                "⚾ Bateador": p_name,
                "👕 Equipo": team_name,
                "🏟️ Condición": condicion,
                "📊 AVG Temp.": f"{season_hits / season_ab:.3f}" if season_ab > 0 else ".000",
                "🔥 AVG L10": f"{l10_hits / l10_ab:.3f}" if l10_ab > 0 else ".000",
                "🎯 Prob. 1+ Hit": f"{prob_1hit_pct}%",   # nombre de columna cambiado
                "📝 Evaluación": eval_str,
                "score": prob_1hit
            })

        resultados.sort(key=lambda x: x['score'], reverse=True)
        top4 = resultados[:4]
        for r in top4:
            del r['score']
        return top4
    except Exception:
        return []

def get_strikeout_hunters(fecha_hoy):
    try:
        juegos_hoy = statsapi.schedule(date=fecha_hoy, sportId=1)
        if not juegos_hoy: return []
        
        pitchers_data = []
        for juego in juegos_hoy:
            if juego.get('status', '') in ['Postponed', 'Cancelled']: continue
            g_status = juego.get('status', '')
            p_local, p_visita = get_starting_pitchers(juego)
            
            matchups = [
                (p_local, juego.get('home_name'), juego.get('away_id'), juego.get('away_name')),
                (p_visita, juego.get('away_name'), juego.get('home_id'), juego.get('home_name'))
            ]
            
            for p_name, p_team, opp_id, opp_name in matchups:
                if not p_name or p_name == 'TBD': continue
                players = statsapi.lookup_player(p_name)
                if not players: continue
                p_id = players[0]['id']
                
                raw_data = statsapi.get('people', {'personIds': p_id, 'hydrate': 'stats(group=[pitching],type=[gameLog])'})
                stats_blocks = raw_data.get('people', [{}])[0].get('stats', [])
                
                l7_ks = 0; l7_outs = 0; juegos_lanzados = 0; ks_hoy_real = 0
                for block in stats_blocks:
                    if block.get('type', {}).get('displayName') == 'gameLog':
                        splits = block.get('splits', [])
                        
                        valid_splits = [s for s in splits if s.get('date', '') < fecha_hoy]
                        valid_splits.sort(key=lambda x: x.get('date', ''), reverse=True)
                        last_7 = valid_splits[:7]
                        juegos_lanzados = len(last_7)
                        
                        for game in last_7:
                            g_stats = game.get('stat', {})
                            l7_ks += int(g_stats.get('strikeOuts', 0))
                            ip_str = str(g_stats.get('inningsPitched', '0.0'))
                            if '.' in ip_str:
                                full, frac = ip_str.split('.')
                                l7_outs += (int(full) * 3) + int(frac)
                            else: l7_outs += int(ip_str) * 3
                                
                        ks_hoy_real = 0
                        outs_hoy_real = 0   # nuevo
                        for s in splits:
                            if s.get('date') == fecha_hoy:
                                gs = s.get('stat', {})
                                ks_hoy_real += int(gs.get('strikeOuts', 0))
                                ip_str = str(gs.get('inningsPitched', '0.0'))
                                if '.' in ip_str:
                                    full, frac = ip_str.split('.')
                                    outs_hoy_real += (int(full) * 3) + int(frac)
                                else:
                                    outs_hoy_real += int(ip_str) * 3
                                
                if juegos_lanzados == 0 or l7_outs == 0: continue
                avg_k_per_start = l7_ks / juegos_lanzados
                
                team_raw = statsapi.get('teams', {'teamId': opp_id, 'hydrate': 'stats(group=[hitting],type=[season,gameLog])'})
                opp_ks = 0; opp_pa = 1
                try:
                    t_stats_blocks = team_raw['teams'][0].get('teamStats', [])
                    ks_equipo_hoy = 0; pa_equipo_hoy = 0
                    
                    for b in t_stats_blocks:
                        if b.get('type', {}).get('displayName') == 'season':
                            t_stats = b['splits'][0]['stat']
                            opp_ks = int(t_stats.get('strikeOuts', 0))
                            opp_pa = int(t_stats.get('plateAppearances', 1))
                        elif b.get('type', {}).get('displayName') == 'gameLog':
                            for t_game in b.get('splits', []):
                                if t_game.get('date') == fecha_hoy:
                                    ks_equipo_hoy += int(t_game.get('stat', {}).get('strikeOuts', 0))
                                    pa_equipo_hoy += int(t_game.get('stat', {}).get('plateAppearances', 0))
                    
                    opp_ks = max(0, opp_ks - ks_equipo_hoy)
                    opp_pa = max(1, opp_pa - pa_equipo_hoy)
                except: pass
                    
                opp_k_pct = opp_ks / opp_pa if opp_pa > 1 else 0.225
                proj_k = avg_k_per_start * (opp_k_pct / 0.225)
                
                proj_k_rounded = round(proj_k, 3) 
                meta_ks = int(round(proj_k))
                
                # Nuevo: probabilidad de alcanzar o superar la proyección (Poisson)
                if proj_k > 0:
                    prob_meta = 1 - poisson.cdf(meta_ks - 1, proj_k)
                else:
                    prob_meta = 0.0
                prob_meta_pct = int(round(prob_meta * 100))
                
                eval_str = "⏳ Pendiente"
                if g_status in ['Final', 'Game Over']:
                    if outs_hoy_real == 0:          # No lanzó en el partido
                        eval_str = "🚫 No lanzó"
                    else:
                        eval_str = f"✅ Acierto ({ks_hoy_real} Ks)" if ks_hoy_real >= meta_ks else f"❌ Fallo ({ks_hoy_real} Ks)"
                
                pitchers_data.append({
                    "⚾ Abridor": p_name,
                    "👕 Equipo": p_team,
                    "⚔️ Rival": opp_name,
                    "🔥 K/9 (L7)": int(round((l7_ks / (l7_outs / 3.0)) * 9.0)),
                    "🎯 Proy. Ponches": meta_ks, 
                    "📝 Evaluación": eval_str,
                    "score": proj_k_rounded,
                    "prob_meta": prob_meta_pct   # guardamos la probabilidad
                })
                
        pitchers_data.sort(key=lambda x: (x['score'], x['⚾ Abridor']), reverse=True)
        top_4 = pitchers_data[:4]
        for r in top_4:
            r["🎯 Proy. Ponches"] = f"{r['🎯 Proy. Ponches']} Ks"
            r["🎲 Prob. Alcanzar Proy."] = f"{r['prob_meta']}%"   # añadimos columna
            del r['score'], r['prob_meta']
        return top_4
    except Exception: return []

# --- INICIALIZACIÓN Y CONTROL DEL TIEMPO (MEDIANOCHE ET) ---
if 'df_mlb' not in st.session_state: st.session_state.df_mlb = None

st.sidebar.markdown("### 🗓️ Motor de Tiempo")
st.sidebar.markdown("Las fechas cambian estrictamente a las 12:00 AM Hora del Este (ET). Selecciona días pasados para auditar el rendimiento del radar.")

tz_et = 'America/New_York'
hoy_et = pd.Timestamp.now(tz_et).date()

fecha_sel = st.sidebar.date_input("Fecha de Análisis:", hoy_et)
st.session_state.fecha_hoy = fecha_sel.strftime('%Y-%m-%d')
st.sidebar.markdown("---")

# --- BARRA LATERAL: MOTOR DE DATOS ---
st.sidebar.markdown("### 📥 Sincronización")
anio_sel = st.sidebar.selectbox("Temporada a procesar:", [2026, 2025, 2024])

if st.sidebar.button("🔄 Descargar Historial Base", type="primary"):
    with st.spinner("Actualizando base de datos central..."):
        try:
            fechas = []
            for m in range(3, 12):
                last_day = calendar.monthrange(anio_sel, m)[1]
                fechas.append((f"{anio_sel}-{m:02d}-01", f"{anio_sel}-{m:02d}-{last_day}"))
            data_total = []
            for start, end in fechas:
                intentos = 0; exito = False
                while intentos < 3 and not exito:
                    try:
                        batch = statsapi.schedule(start_date=start, end_date=end, sportId=1)
                        if batch: data_total.extend(batch)
                        exito = True
                        time.sleep(1.0)
                    except Exception:
                        intentos += 1; time.sleep(2.0)
            df_full = pd.DataFrame(data_total)
            df_full = df_full[df_full['status'].isin(['Final', 'Game Over'])].copy()
            if 'game_type' in df_full.columns:
                df_full = df_full[df_full['game_type'].isin(['R', 'P'])]
            df_full = df_full[['home_name', 'away_name', 'home_score', 'away_score', 'game_date']]
            df_full.columns = ['Local', 'Visitante', 'Carreras_Local', 'Carreras_Visitante', 'Date']
            df_full = df_full[df_full['Local'].isin(MLB_TEAM_WHITELIST) & df_full['Visitante'].isin(MLB_TEAM_WHITELIST)]
            
            elo_dict = {team: 1500.0 for team in MLB_TEAM_WHITELIST}
            h_elo_l, h_elo_v = [], []
            for _, row in df_full.iterrows():
                l, v = row['Local'], row['Visitante']
                el, ev = elo_dict[l], elo_dict[v]
                h_elo_l.append(el); h_elo_v.append(ev)
                diff = 1 / (1 + 10 ** ((ev - el) / 400))
                res = 1.0 if row['Carreras_Local'] > row['Carreras_Visitante'] else 0.0
                elo_dict[l] += 6 * (res - diff)
                elo_dict[v] += 6 * ((1 - res) - (1 - diff))
            
            df_full['Elo_L'], df_full['Elo_V'] = h_elo_l, h_elo_v
            st.session_state.df_mlb = df_full
            st.sidebar.success("✅ Base de datos al día.")
        except Exception as e: st.sidebar.error(f"Error Crítico: {e}")

# --- ÁREA PRINCIPAL ---
if st.session_state.df_mlb is not None:
    df_historico = st.session_state.df_mlb.copy()
    df_filtrado = df_historico[df_historico['Date'] < st.session_state.fecha_hoy].copy()
    
    if len(df_filtrado) > 0:
        df_filtrado['Win'] = (df_filtrado['Carreras_Local'] > df_filtrado['Carreras_Visitante']).astype(int)
        clf = RandomForestClassifier(max_depth=MAX_DEPTH_ELO, random_state=42).fit(df_filtrado[['Elo_L', 'Elo_V']], df_filtrado['Win'])
    
    # --- ACTUALIZACIÓN DE PESTAÑAS (Agregada tab 4) ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Cartelera del Día",
    "💣 Caza-Jonrones",
    "🔥 Caza-Ponches",
    "🔹 Caza-Hits (2+ Hits)",
    "🧮 Calculadora +EV"
])
    
    with tab1:
        st.markdown(f"### 🎯 Partidos programados para el: **{st.session_state.fecha_hoy}**")

        # Botón para generar/actualizar predicciones (siempre visible)
        if st.button("⚡ Analizar y Evaluar Cartelera", type="primary", use_container_width=True):
            if len(df_filtrado) == 0:
                st.error("No hay suficientes datos históricos previos a esta fecha para entrenar el modelo.")
                st.session_state.resultados_jornada = None
            else:
                with st.spinner("Escaneando predicciones ciegas y verificando con resultados reales..."):
                    juegos_hoy = statsapi.schedule(date=st.session_state.fecha_hoy, sportId=1)

                    if not juegos_hoy:
                        st.warning("No hay juegos programados para esta fecha.")
                        st.session_state.resultados_jornada = None
                    else:
                        resultados_jornada = []
                        equipos_procesados = set()

                        for juego in juegos_hoy:
                            estado_juego = juego.get('status', '')
                            estados_validos = ['Scheduled', 'Pre-Game', 'Warmup', 'Delayed Start', 'In Progress', 'Final', 'Game Over']
                            if estado_juego not in estados_validos: continue

                            e_local = juego['home_name']
                            e_visita = juego['away_name']
                            p_local, p_visita = get_starting_pitchers(juego)

                            if e_local not in MLB_TEAM_WHITELIST or e_visita not in MLB_TEAM_WHITELIST: continue
                            if e_local in equipos_procesados or e_visita in equipos_procesados: continue

                            equipos_procesados.add(e_local)
                            equipos_procesados.add(e_visita)

                            game_dt_str = juego.get('game_datetime', '')
                            if game_dt_str:
                                try:
                                    dt = pd.to_datetime(game_dt_str)
                                    if dt.tzinfo is None: dt = dt.tz_localize('UTC')
                                    hora_et = dt.tz_convert('America/New_York').strftime('%I:%M %p')
                                except: hora_et = 'TBD'
                            else: hora_et = 'TBD'

                            rec_l = get_team_record(e_local, df_filtrado)
                            rec_v = get_team_record(e_visita, df_filtrado)

                            elo_l = df_filtrado[df_filtrado['Local'] == e_local].tail(1)['Elo_L'].values[0] if len(df_filtrado[df_filtrado['Local'] == e_local]) > 0 else 1500
                            elo_v = df_filtrado[df_filtrado['Visitante'] == e_visita].tail(1)['Elo_V'].values[0] if len(df_filtrado[df_filtrado['Visitante'] == e_visita]) > 0 else 1500
                            elo_l += 35

                            racha_l = get_recent_form(e_local, df_filtrado)
                            racha_v = get_recent_form(e_visita, df_filtrado)
                            h2h = get_h2h_wins(e_local, e_visita, df_filtrado)
                            luck_l = get_pythagorean_luck(e_local, df_filtrado)
                            luck_v = get_pythagorean_luck(e_visita, df_filtrado)
                            split_l, split_v = get_splits_win_pct(e_local, e_visita, df_filtrado)

                            whip_l = get_pitcher_whip(p_local, st.session_state.fecha_hoy)
                            whip_v = get_pitcher_whip(p_visita, st.session_state.fecha_hoy)

                            prob = clf.predict_proba(np.array([[elo_l, elo_v]]))[0][1]
                            pitcher_adj = (whip_v - whip_l) * 0.15

                            prob_final_local = (prob + (racha_l - racha_v)*PESO_RACHA + (h2h - 0.5)*PESO_H2H +
                                                (luck_l - luck_v)*PESO_PITAGORICO + (split_l - split_v)*PESO_SPLITS + pitcher_adj)

                            ganador = e_local if prob_final_local > 0.5 else e_visita
                            pct_bruto = prob_final_local if prob_final_local > 0.5 else 1.0 - prob_final_local
                            pct_final = int(round(max(min(pct_bruto, 0.99), 0.01) * 100))

                            jugada_str = f"{ganador} (A Ganar)"
                            prob_str = f"{pct_final}%"
                            score_val = pct_final

                            eval_str = "⏳ Pendiente"
                            if estado_juego in ['Final', 'Game Over']:
                                r_local = juego.get('home_score', 0)
                                r_visita = juego.get('away_score', 0)
                                r_ganador = e_local if r_local > r_visita else e_visita
                                marcador_str = f"({r_local}-{r_visita})"
                                eval_str = f"✅ Acierto {marcador_str}" if r_ganador == ganador else f"❌ Fallo {marcador_str}"

                            resultados_jornada.append({
                                "⏰ Hora (ET)": hora_et,
                                "✈️ Visitante": f"{e_visita} ({rec_v})",
                                "🏠 Local": f"{e_local} ({rec_l})",
                                "⚾ Abridor (V)": f"{p_visita or 'TBD'} ({whip_v:.2f})",
                                "⚾ Abridor (L)": f"{p_local or 'TBD'} ({whip_l:.2f})",
                                "🎯 Jugada Recomendada": jugada_str,
                                "📊 Prob.": prob_str,
                                "📝 Evaluación": eval_str,
                                "raw_time": game_dt_str or "9999-12-31T23:59:59Z",
                                "score": score_val
                            })

                        resultados_jornada.sort(key=lambda x: x['raw_time'])
                        st.session_state.resultados_jornada = resultados_jornada
                        st.success("✅ Análisis completado. Resultados guardados.")

        # --- Mostrar resultados almacenados (persiste al cambiar de pestaña) ---
        if "resultados_jornada" in st.session_state and st.session_state.resultados_jornada is not None:
            resultados_jornada = st.session_state.resultados_jornada
            df_resultados = pd.DataFrame(resultados_jornada).drop(columns=['score', 'raw_time'], errors='ignore')

            def color_whip(row):
                styles = [''] * len(row)
                for j, col in enumerate(row.index):
                    if col in ['⚾ Abridor (V)', '⚾ Abridor (L)']:
                        try:
                            whip = float(str(row[col]).split('(')[-1].replace(')', ''))
                            if whip < 1.00: styles[j] = 'color: #00cc66; font-weight: bold;'
                            elif whip <= 1.30: styles[j] = 'color: #ff9900; font-weight: bold;'
                            else: styles[j] = 'color: #ff4d4d; font-weight: bold;'
                        except: pass
                return styles

            df_estilizado = df_resultados.style.apply(color_whip, axis=1)\
                .set_properties(**{'text-align': 'center'})\
                .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

            st.dataframe(df_estilizado, use_container_width=True, hide_index=True)

            total_evaluados = sum(1 for e in df_resultados['📝 Evaluación'] if '✅' in e or '❌' in e)
            aciertos = sum(1 for e in df_resultados['📝 Evaluación'] if '✅' in e)

            if total_evaluados > 0:
                efectividad = (aciertos / total_evaluados) * 100
                st.markdown("### 📊 Rendimiento de la Cartelera")
                col1, col2, col3 = st.columns(3)
                col1.metric("Pronósticos Finalizados", total_evaluados)
                col2.metric("Aciertos Confirmados", aciertos)
                col3.metric("Efectividad del Radar", f"{int(round(efectividad))}%")
            else:
                st.info("Aún no hay juegos finalizados para calcular la efectividad de la jornada.")
        else:
            st.info("Presiona el botón 'Analizar y Evaluar Cartelera' para generar las predicciones del día.")

    with tab2:
        st.markdown("### 💣 Radar de Jonrones: Filtro de Regresión + Localía")
        if st.button("🔍 Escanear Mercado de Jonrones (Top 4 Limpio)", type="primary", use_container_width=True):
            with st.spinner("Evaluando rachas y buscando jonrones reales de la jornada..."):
                resultados_hr = get_hr_hunters(anio_sel, st.session_state.fecha_hoy)
                if resultados_hr:
                    st.session_state[f"resultados_hr_{st.session_state.fecha_hoy}"] = resultados_hr
                else:
                    st.session_state[f"resultados_hr_{st.session_state.fecha_hoy}"] = None
                    st.warning("No se detectaron líderes válidos o datos para esta fecha.")

        clave_hr = f"resultados_hr_{st.session_state.fecha_hoy}"
        if clave_hr in st.session_state and st.session_state[clave_hr] is not None:
            df_hr = pd.DataFrame(st.session_state[clave_hr])
            df_hr_estilizado = df_hr.style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
            st.dataframe(df_hr_estilizado, use_container_width=True, hide_index=True)

            total_evaluados = sum(1 for e in df_hr['📝 Evaluación'] if '✅' in e or '❌' in e)
            aciertos = sum(1 for e in df_hr['📝 Evaluación'] if '✅' in e)

            if total_evaluados > 0:
                efectividad = (aciertos / total_evaluados) * 100
                st.markdown("### 📊 Rendimiento Caza-Jonrones")
                c1, c2, c3 = st.columns(3)
                c1.metric("Bateadores Evaluados", total_evaluados)
                c2.metric("Jonrones Acertados", aciertos)
                c3.metric("Efectividad", f"{int(round(efectividad))}%")
        elif clave_hr not in st.session_state:
            st.info("Presiona el botón para escanear el mercado de jonrones.")

    with tab3:
        st.markdown("### 🔥 Radar de Ponches: Pitcher K/9 vs Vulnerabilidad del Rival")
        if st.button("🎯 Cazar Ponches del Día (Top 4)", type="primary", use_container_width=True):
            with st.spinner("Haciendo el cruce de vulnerabilidad y auditando ponches finales..."):
                resultados_k = get_strikeout_hunters(st.session_state.fecha_hoy)
                if resultados_k:
                    st.session_state[f"resultados_k_{st.session_state.fecha_hoy}"] = resultados_k
                else:
                    st.session_state[f"resultados_k_{st.session_state.fecha_hoy}"] = None
                    st.warning("No hay suficientes datos de pitcheo para evaluar esta jornada.")

        clave_k = f"resultados_k_{st.session_state.fecha_hoy}"
        if clave_k in st.session_state and st.session_state[clave_k] is not None:
            df_k = pd.DataFrame(st.session_state[clave_k])
            df_k_estilizado = df_k.style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
            st.dataframe(df_k_estilizado, use_container_width=True, hide_index=True)

            total_evaluados = sum(1 for e in df_k['📝 Evaluación'] if '✅' in e or '❌' in e)
            aciertos = sum(1 for e in df_k['📝 Evaluación'] if '✅' in e)

            if total_evaluados > 0:
                efectividad = (aciertos / total_evaluados) * 100
                st.markdown("### 📊 Rendimiento Caza-Ponches")
                c1, c2, c3 = st.columns(3)
                c1.metric("Lanzadores Evaluados", total_evaluados)
                c2.metric("Metas Superadas", aciertos)
                c3.metric("Efectividad", f"{int(round(efectividad))}%")
        elif clave_k not in st.session_state:
            st.info("Presiona el botón para cazar ponches del día.")
                
    with tab4:
        st.markdown("### 🔹 Radar de Hits: Probabilidad de 1+ Imparables")
        if st.button("🔎 Buscar Bateadores con 1+ Hits (Top 4)", type="primary", use_container_width=True):
            with st.spinner("Calculando probabilidades de dar hits..."):
                resultados_hits = get_hit_hunters(anio_sel, st.session_state.fecha_hoy)
                if resultados_hits:
                    st.session_state[f"resultados_hits_{st.session_state.fecha_hoy}"] = resultados_hits
                else:
                    st.session_state[f"resultados_hits_{st.session_state.fecha_hoy}"] = None
                    st.warning("No se encontraron bateadores con datos suficientes para hoy.")

        clave_hits = f"resultados_hits_{st.session_state.fecha_hoy}"
        if clave_hits in st.session_state and st.session_state[clave_hits] is not None:
            df_hits = pd.DataFrame(st.session_state[clave_hits])
            df_hits_estilizado = df_hits.style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
            st.dataframe(df_hits_estilizado, use_container_width=True, hide_index=True)

            total_evaluados = sum(1 for e in df_hits['📝 Evaluación'] if '✅' in e or '❌' in e)
            aciertos = sum(1 for e in df_hits['📝 Evaluación'] if '✅' in e)

            if total_evaluados > 0:
                efectividad = (aciertos / total_evaluados) * 100
                st.markdown("### 📊 Rendimiento Caza-Hits (2+ Imparables)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Bateadores Evaluados", total_evaluados)
                c2.metric("Aciertos (2+ Hits)", aciertos)
                c3.metric("Efectividad", f"{int(round(efectividad))}%")
        elif clave_hits not in st.session_state:
            st.info("Presiona el botón para buscar bateadores con alta probabilidad de 2+ hits.")

    with tab5:
        st.markdown("### 🧮 Calculadora de Valor Esperado (+EV)")
        st.markdown("Compara la probabilidad matemática del Radar con la cuota decimal de tu casa de apuestas para descubrir si la jugada es rentable a largo plazo.")
        
        c1, c2 = st.columns(2)
        with c1:
            prob_radar = st.number_input("📊 Probabilidad que arrojó el Radar (%)", min_value=1, max_value=99, value=55, step=1)
        with c2:
            cuota_decimal = st.number_input("🏦 Cuota Decimal (ej. 1.91, 2.50)", min_value=1.01, value=1.91, step=0.01)
            
        if cuota_decimal > 1.0:
            # Probabilidad implícita a partir de la cuota decimal
            prob_implicita = 1.0 / cuota_decimal
            
            prob_radar_dec = prob_radar / 100.0
            # Valor esperado
            ev_pct = (prob_radar_dec * cuota_decimal) - 1.0
            
            # Redondeo a enteros para mostrar
            prob_implicita_int = int(round(prob_implicita * 100))
            ev_pct_int = int(round(ev_pct * 100))
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Probabilidad que exige el Casino", f"{prob_implicita_int}%")
            
            # ---- INDICADOR VISUAL SEGÚN EV ----
            if ev_pct >= 0.10:
                st.toast('Apuesta de alto valor (+10% EV)', icon='💎')
                st.success("💎 Apuesta de alto valor (+10% EV)")
            elif ev_pct >= 0.05:
                st.success("✅ Apuesta recomendada (+5% EV)")
            elif ev_pct > 0:
                st.warning("⚠️ Valor marginal, solo si el radar es muy fiable")
            else:
                st.error("❌ Apuesta no rentable")
            
            # Texto explicativo
            if ev_pct > 0:
                col2.metric("Valor Esperado (EV)", f"+{ev_pct_int}%", "Rentable (+EV)")
                st.success(f"✅ **¡Apuesta de Valor!** El radar le da **{prob_radar}%** de probabilidad de éxito, y la casa de apuestas te está cobrando como si solo tuviera **{prob_implicita_int}%**. Tienes ventaja matemática. Si repites esta apuesta 100 veces, ganarás dinero.")
            else:
                col2.metric("Valor Esperado (EV)", f"{ev_pct_int}%", "No Rentable (-EV)", delta_color="inverse")
                st.error(f"❌ **Déjala Pasar.** El casino está protegiendo su dinero exigiendo un **{prob_implicita_int}%** de éxito, pero el radar solo le da un **{prob_radar}%**. A largo plazo, esta apuesta te hará perder tu capital (bankroll).")
