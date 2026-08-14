import streamlit as st
import pandas as pd
import numpy as np
import statsapi
import time
import datetime
import calendar
import io  
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import poisson
from scipy.stats import binom

st.set_page_config(page_title="Predicción MLB Automatizada", layout="wide", page_icon="⚾")

st.title("⚾ Predicción MLB: Radar Diario Automatizado")
st.markdown("Proyección Sabermétrica")
st.markdown("---")

MAX_DEPTH_ELO = 5               

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

def get_bullpen_metrics(team_id, fecha_corte):
    avg_whip = 1.30 
    if not team_id: return avg_whip
    
    try:
        dt_corte = datetime.datetime.strptime(fecha_corte, '%Y-%m-%d')
        dt_7d = dt_corte - datetime.timedelta(days=7)
        dt_3d = dt_corte - datetime.timedelta(days=3)
        
        str_7d = dt_7d.strftime('%Y-%m-%d')
        str_3d = dt_3d.strftime('%Y-%m-%d')
        
        juegos = statsapi.schedule(team=team_id, start_date=str_7d, end_date=fecha_corte)
        if not juegos: return avg_whip
        
        total_hits_7d = 0; total_bb_7d = 0; total_outs_7d = 0
        bp_outs_3d = 0
        
        for juego in juegos:
            if juego.get('status') not in ['Final', 'Game Over']: continue
            
            game_date = juego.get('game_date', '')
            game_id = juego.get('game_id')
            if not game_id: continue
            
            try:
                box = statsapi.boxscore_data(game_id)
                team_side = 'home' if juego.get('home_id') == team_id else 'away'
                
                # Lista de TODOS los lanzadores del equipo en ESE juego
                pitchers_list = box.get(team_side, {}).get('pitchers', [])
                
                # Si solo hay 1 lanzador, fue un juego completo (no hubo bullpen)
                if len(pitchers_list) <= 1: continue
                
                # AISLAMOS EL BULLPEN: Ignoramos al [0] (el abridor)
                relevistas = pitchers_list[1:]
                
                outs_juego_bp = 0
                
                for pid in relevistas:
                    p_key = f"ID{pid}"
                    p_stats = box.get(team_side, {}).get('players', {}).get(p_key, {}).get('stats', {}).get('pitching', {})
                    
                    if not p_stats: continue
                        
                    hits = int(p_stats.get('hits', 0))
                    bb = int(p_stats.get('baseOnBalls', 0))
                    ip_str = str(p_stats.get('inningsPitched', '0.0'))
                    
                    if '.' in ip_str:
                        full, frac = ip_str.split('.')
                        outs = (int(full) * 3) + int(frac)
                    else:
                        outs = int(ip_str) * 3
                        
                    total_hits_7d += hits
                    total_bb_7d += bb
                    total_outs_7d += outs
                    outs_juego_bp += outs
                    
                # Fatiga: Sumar solo carga real del bullpen en los últimos 3 días
                if str_3d <= game_date < fecha_corte:
                    bp_outs_3d += outs_juego_bp
                    
            except: pass
                
        if total_outs_7d == 0: return avg_whip
        
        # WHIP EXCLUSIVO DE RELEVISTAS
        whip_7d = (total_hits_7d + total_bb_7d) / (total_outs_7d / 3.0)
        
        # Penalización por Fatiga Real: Si el bullpen lanzó más de 30 outs (~10 innings) en 3 días.
        fatigue_mod = 1.0
        if bp_outs_3d > 30:
            fatigue_mod = 1.0 + ((bp_outs_3d - 30) * 0.008) 
            
        final_whip = round(whip_7d * fatigue_mod, 2)
        # Tolerancia ampliada: Los relevistas pueden ser un desastre total o intocables
        return max(0.80, min(3.00, final_whip))
    except Exception: 
        return avg_whip

def get_hit_hunters(anio, fecha_hoy):
    try:
        juegos_hoy = statsapi.schedule(date=fecha_hoy, sportId=1)
        equipos_hoy = {}
        
        # Función rápida interna para saber si el pitcher es Zurdo (L) o Derecho (R)
        def get_pitcher_hand(p_name):
            if not p_name or p_name == 'TBD': return 'R' # Por defecto asumimos derecho
            try:
                pl = statsapi.lookup_player(p_name)
                return pl[0].get('pitchHand', {}).get('code', 'R') if pl else 'R'
            except: return 'R'

        for juego in juegos_hoy:
            if juego.get('status', '') not in ['Postponed', 'Cancelled']:
                p_local, p_visita = get_starting_pitchers(juego)
                h_id = juego.get('home_id')
                a_id = juego.get('away_id')
                
                # Extraemos la mano de los abridores de hoy
                hand_local = get_pitcher_hand(p_local)
                hand_visita = get_pitcher_hand(p_visita)
                
                equipos_hoy[h_id] = {
                    'condicion': 'Local', 'status': juego.get('status'),
                    'opp_id': a_id, 'opp_pitcher': p_visita, 'opp_hand': hand_visita
                }
                equipos_hoy[a_id] = {
                    'condicion': 'Visitante', 'status': juego.get('status'),
                    'opp_id': h_id, 'opp_pitcher': p_local, 'opp_hand': hand_local
                }

        data = statsapi.get('stats_leaders', {'leaderCategories': 'battingAverage', 'season': anio, 'limit': 80, 'statGroup': 'hitting'})
        if not data or 'leagueLeaders' not in data or len(data['leagueLeaders']) == 0:
            return []

        leaders = data['leagueLeaders'][0].get('leaders', [])
        jugadores_activos = []
        for p in leaders:
            team_id = p.get('team', {}).get('id')
            if team_id in equipos_hoy:
                p['condicion_hoy'] = equipos_hoy[team_id]['condicion']
                p['game_status'] = equipos_hoy[team_id]['status']
                p['opp_id'] = equipos_hoy[team_id]['opp_id']
                p['opp_pitcher'] = equipos_hoy[team_id]['opp_pitcher']
                p['opp_hand'] = equipos_hoy[team_id]['opp_hand']
                p['team_name'] = p.get('team', {}).get('name', 'Unknown')
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
            
            opp_id = p.get('opp_id')
            opp_pitcher = p.get('opp_pitcher')
            opp_hand = p.get('opp_hand')

            raw_data = statsapi.get('people', {'personIds': p_id, 'hydrate': 'currentTeam,stats(group=[hitting],type=[season,gameLog])'})
            person = raw_data.get('people', [{}])[0]
            
            current_team_obj = person.get('currentTeam', {})
            if current_team_obj:
                team_name = current_team_obj.get('name', team_name)
                
            # Extraemos la mano del Bateador: 'L' (Zurdo), 'R' (Derecho), 'S' (Switch/Ambidiestro)
            bat_hand = person.get('batSide', {}).get('code', 'R')
                
            stats_blocks = person.get('stats', [])

            season_ab = 1; season_hits = 0
            l10_hits = 0; l10_ab = 0
            hits_hoy_real = 0; ab_hoy_real = 0

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
                        l10_hits += int(g_stats.get('hits', 0))
                        l10_ab += int(g_stats.get('atBats', 0))

                    for game in splits:
                        if game.get('date') == fecha_hoy:
                            hits_hoy_real += int(game.get('stat', {}).get('hits', 0))
                            ab_hoy_real += int(game.get('stat', {}).get('atBats', 0))

            if game_status in ['Final', 'Game Over'] and ab_hoy_real == 0: continue

            season_hits = max(0, season_hits - hits_hoy_real)
            season_ab = max(1, season_ab - ab_hoy_real)
            l10_ab = max(1, l10_ab)

            # 1. Promedio base
            avg_index_base = (season_hits / season_ab * 0.6) + (l10_hits / l10_ab * 0.4)

            # 2. SPLITS: Ventaja de Pelotón (Platoon Advantage)
            platoon_mod = 1.0 # Neutro
            if bat_hand == 'S': 
                platoon_mod = 1.05 # Ambidiestro siempre tiene ventaja
            elif bat_hand != opp_hand:
                platoon_mod = 1.05 # Manos contrarias (Ej: Bateador Zurdo vs Pitcher Derecho)
            else:
                platoon_mod = 0.95 # Misma mano (Ej: Bateador Derecho vs Pitcher Derecho)

            # 3. Evaluación del Pitcheo Rival
            whip_abridor = get_pitcher_whip(opp_pitcher, fecha_hoy)
            whip_bullpen = get_bullpen_metrics(opp_id, fecha_hoy)
            whip_combinado = (whip_abridor * 0.6) + (whip_bullpen * 0.4)
            factor_pitcheo = whip_combinado / 1.30
            factor_pitcheo = max(0.90, min(1.10, factor_pitcheo))
            
            # 4. Ajuste Final: Promedio × Factor Pitcheo × Ventaja de Pelotón
            avg_index_ajustado = avg_index_base * factor_pitcheo * platoon_mod

            # 5. Cálculo Binomial Realista
            prob_1hit = 1 - (1 - avg_index_ajustado) ** 3.8
            prob_1hit_pct = int(round(prob_1hit * 100))

            eval_str = "⏳ Pendiente"
            if game_status in ['Final', 'Game Over']:
                if ab_hoy_real == 0:
                    eval_str = "🚫 No jugó"
                else:
                    eval_str = "✅ Acierto" if hits_hoy_real >= 1 else "❌ Fallo"

            # 🌟 FILTRO DE ALTA SEGURIDAD (HITS) 🌟
            avg_l10_val = l10_hits / l10_ab if l10_ab > 0 else 0
            
            es_alta_seg = False
            # Regla 1, 2 y 3: Alta prob (>80%), Racha caliente (>.300), Pitcheo Global vulnerable (WHIP Combinado >= 1.30)
            if prob_1hit_pct >= 80 and avg_l10_val >= 0.300 and whip_combinado >= 1.30:
                # Regla 4: Ventaja de Pelotón (Manos opuestas o Switch Hitter)
                if bat_hand == 'S' or bat_hand != opp_hand:
                    es_alta_seg = True

            nombre_bateador = f"⭐ {p_name}" if es_alta_seg else p_name

            resultados.append({
                "⚾ Bateador": nombre_bateador,
                "👕 Equipo": team_name,
                "⚔️ Rival": f"{opp_pitcher or 'TBD'} ({opp_hand})",
                "🏟️ Condición": condicion,
                "📊 AVG Temp.": f"{season_hits / season_ab:.3f}" if season_ab > 0 else ".000",
                "🔥 AVG L10": f"{l10_hits / l10_ab:.3f}" if l10_ab > 0 else ".000",
                "🎯 Prob. 1+ Hit": f"{prob_1hit_pct}%",
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
            
            # 1. AGREGAMOS EL ID DEL EQUIPO AL MATCHUP PARA EL FILTRO ANTI-CLONES
            matchups = [
                (p_local, juego.get('home_id'), juego.get('home_name'), juego.get('away_id'), juego.get('away_name')),
                (p_visita, juego.get('away_id'), juego.get('away_name'), juego.get('home_id'), juego.get('home_name'))
            ]
            
            for p_name, team_id, p_team, opp_id, opp_name in matchups:
                if not p_name or p_name == 'TBD': continue
                players = statsapi.lookup_player(p_name)
                if not players: continue
                
                # -------------------------------------------------------------
                # 2. FILTRO ANTI-HOMÓNIMOS
                p_id = None
                raw_data = None
                
                for pl in players:
                    temp_id = pl['id']
                    # Extraemos el equipo actual del jugador
                    temp_data = statsapi.get('people', {'personIds': temp_id, 'hydrate': 'currentTeam,stats(group=[pitching],type=[gameLog])'})
                    person_info = temp_data.get('people', [{}])[0]
                    
                    # Verificamos si pertenece al equipo que juega hoy
                    if person_info.get('currentTeam', {}).get('id') == team_id:
                        p_id = temp_id
                        raw_data = temp_data
                        break
                        
                # Si no lo encontró, usa el primero por defecto
                if not p_id or not raw_data:
                    p_id = players[0]['id']
                    raw_data = statsapi.get('people', {'personIds': p_id, 'hydrate': 'currentTeam,stats(group=[pitching],type=[gameLog])'})
                # -------------------------------------------------------------
                
                stats_blocks = raw_data.get('people', [{}])[0].get('stats', [])
                
                all_splits = []
                for block in stats_blocks:
                    if block.get('type', {}).get('displayName') == 'gameLog':
                        all_splits.extend(block.get('splits', []))
                
                valid_splits = [s for s in all_splits if s.get('date', '') < fecha_hoy]
                valid_splits.sort(key=lambda x: x.get('date', ''), reverse=True)
                
                last_7 = valid_splits[:7]
                juegos_lanzados = len(last_7)
                
                l7_ks = 0; l7_outs = 0; ks_hoy_real = 0; outs_hoy_real = 0
                ks_list = []
                
                for game in last_7:
                    g_stats = game.get('stat', {})
                    k = int(g_stats.get('strikeOuts', 0))
                    l7_ks += k
                    ks_list.append(k) # Imprescindible para calcular la mediana
                    ip_str = str(g_stats.get('inningsPitched', '0.0'))
                    if '.' in ip_str:
                        full, frac = ip_str.split('.')
                        l7_outs += (int(full) * 3) + int(frac)
                    else: 
                        l7_outs += int(ip_str) * 3
                        
                for s in all_splits:
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

                ks_sorted = sorted(ks_list)
                median_k = ks_sorted[len(ks_sorted)//2] if ks_sorted else 0

                anio_str = str(fecha_hoy)[:4]
                team_raw = statsapi.get('teams', {
                    'teamId': opp_id, 
                    'season': anio_str,
                    'hydrate': f'teamStats(group=[hitting],type=[season,gameLog],season={anio_str})'
                })
                
                season_ks = 0; season_pa = 1
                l10_ks = 0; l10_pa = 0
                
                try:
                    t_stats_blocks = team_raw['teams'][0].get('teamStats', [])
                    ks_equipo_hoy = 0; pa_equipo_hoy = 0
                    
                    for b in t_stats_blocks:
                        if b.get('type', {}).get('displayName') == 'season':
                            t_stats = b['splits'][0]['stat']
                            season_ks = int(t_stats.get('strikeOuts', 0))
                            season_pa = int(t_stats.get('plateAppearances', 1))
                            
                        elif b.get('type', {}).get('displayName') == 'gameLog':
                            splits = b.get('splits', [])
                            valid_splits = [s for s in splits if s.get('date', '') < fecha_hoy]
                            valid_splits.sort(key=lambda x: x.get('date', ''), reverse=True)
                            
                            for t_game in valid_splits[:10]:
                                l10_ks += int(t_game.get('stat', {}).get('strikeOuts', 0))
                                l10_pa += int(t_game.get('stat', {}).get('plateAppearances', 0))
                                
                            for t_game in splits:
                                if t_game.get('date') == fecha_hoy:
                                    ks_equipo_hoy += int(t_game.get('stat', {}).get('strikeOuts', 0))
                                    pa_equipo_hoy += int(t_game.get('stat', {}).get('plateAppearances', 0))
                                    
                    season_ks = max(0, season_ks - ks_equipo_hoy)
                    season_pa = max(1, season_pa - pa_equipo_hoy)
                except: pass
                
                season_k_pct = season_ks / season_pa if season_pa > 1 else 0.225
                l10_k_pct = l10_ks / l10_pa if l10_pa > 1 else season_k_pct
                
                blended_k_pct = (season_k_pct * 0.40) + (l10_k_pct * 0.60)
                factor_rival = blended_k_pct / 0.225
                factor_rival = max(0.80, min(1.20, factor_rival))

                avg_ip = (l7_outs / 3.0) / juegos_lanzados
                factor_ip = min(1.0, avg_ip / 6.0)
                
                avg_outs_redondeado = int(round(l7_outs / juegos_lanzados))
                innings_enteros = avg_outs_redondeado // 3
                outs_sobrantes = avg_outs_redondeado % 3
                
                # -------------------------------------------------------------
                # NUEVO FILTRO DE VOLUMEN: Entre 3.0 y 5.0 innings obligatorios
                # -------------------------------------------------------------
                if innings_enteros < 3 or innings_enteros > 5:
                    continue
                # Si proyecta exactamente 5 innings, no puede tener outs sobrantes (ej. 5.1 o 5.2 se descartan)
                if innings_enteros == 5 and outs_sobrantes > 0:
                    continue
                
                ip_pantalla = f"{innings_enteros}.{outs_sobrantes}"

                proj_k = (median_k * factor_rival * factor_ip)
                proj_k_redondeada = int(round(proj_k))
                
                meta_ks_max = 4 
                prob_under = poisson.cdf(meta_ks_max, proj_k) if proj_k > 0 else 1.0
                prob_under_pct = int(round(prob_under * 100))

                k9 = int(round((l7_ks / (l7_outs / 3.0)) * 9.0))

                eval_str = "⏳ Pendiente"
                if g_status in ['Final', 'Game Over']:
                    if outs_hoy_real == 0:
                        eval_str = "🚫 No lanzó"
                    else:
                        eval_str = f"✅ Acierto (Under: {ks_hoy_real} Ks)" if ks_hoy_real <= meta_ks_max else f"❌ Fallo (Over: {ks_hoy_real} Ks)"

                pitchers_data.append({
                    "⚾ Abridor": p_name,
                    "👕 Equipo": p_team,
                    "⚔️ Rival": opp_name,
                    "⏱️ Proy. IP": ip_pantalla,
                    "❄️ K/9 (L7)": k9,
                    "🎯 Proy. Ponches": proj_k_redondeada,
                    "score": proj_k, 
                    "prob_under_pct": prob_under_pct,
                    "prob_under_exacta": prob_under,
                    "📝 Evaluación": eval_str
                })

        # Ordenamos estrictamente por la probabilidad decimal exacta, de mayor a menor
        pitchers_data.sort(key=lambda x: x['prob_under_exacta'], reverse=True)
        top_4 = pitchers_data[:4]

        nuevo_top4 = []
        for r in top_4:
            # 🌟 FILTRO DE ALTA SEGURIDAD (BAJAS) 🌟
            ip_val = float(r["⏱️ Proy. IP"])
            
            es_alta_seg = False
            if r["prob_under_pct"] >= 85 and ip_val <= 4.0 and r["❄️ K/9 (L7)"] <= 6:
                es_alta_seg = True
                
            nombre_abridor = f"⭐ {r['⚾ Abridor']}" if es_alta_seg else r['⚾ Abridor']

            nuevo_top4.append({
                "⚾ Abridor": nombre_abridor,
                "👕 Equipo": r["👕 Equipo"],
                "⚔️ Rival": r["⚔️ Rival"],
                "⏱️ Proy. IP": r["⏱️ Proy. IP"],
                "❄️ K/9 (L7)": r["❄️ K/9 (L7)"],  # <--- AQUÍ REGRESAMOS LA COLUMNA A LA PANTALLA
                "📉 Prob. Under 4.5": f"{r['prob_under_pct']}%",
                "📝 Evaluación": r["📝 Evaluación"]
            })
        return nuevo_top4
    except Exception:
        return []

def get_detailed_pitcher_stats(pitcher_name, fecha_corte):
    res = {"IP": "0.0", "H": 0, "BB": 0, "K": 0, "ER": 0, "ERA": "4.50", "WHIP": "1.30"}
    if not pitcher_name or pitcher_name == 'TBD': return res
    try:
        players = statsapi.lookup_player(pitcher_name)
        if not players: return res
        player_id = players[0]['id']
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
                            
                            total_outs = 0; total_hits = 0; total_bb = 0; total_k = 0; total_er = 0
                            for game in last_7:
                                g_stats = game.get('stat', {})
                                total_hits += int(g_stats.get('hits', 0))
                                total_bb += int(g_stats.get('baseOnBalls', 0))
                                total_k += int(g_stats.get('strikeOuts', 0))
                                total_er += int(g_stats.get('earnedRuns', 0)) # Carreras Limpias
                                ip_str = str(g_stats.get('inningsPitched', '0.0'))
                                if '.' in ip_str:
                                    full, frac = ip_str.split('.')
                                    total_outs += (int(full) * 3) + int(frac)
                                else:
                                    total_outs += int(ip_str) * 3
                                    
                            if total_outs > 0:
                                ip_calc = total_outs / 3.0
                                res["IP"] = f"{int(total_outs//3)}.{total_outs%3}"
                                res["H"] = total_hits
                                res["BB"] = total_bb
                                res["K"] = total_k
                                res["ER"] = total_er
                                res["ERA"] = f"{(total_er * 9) / ip_calc:.2f}"
                                res["WHIP"] = f"{(total_hits + total_bb) / ip_calc:.2f}"
    except Exception: pass
    return res

def get_detailed_bullpen_stats(team_id, fecha_corte):
    res = {"IP": "0.0", "H": 0, "BB": 0, "K": 0, "ER": 0, "ERA": "0.00", "WHIP": "0.00"}
    if not team_id: return res
    try:
        dt_corte = datetime.datetime.strptime(fecha_corte, '%Y-%m-%d')
        dt_7d = dt_corte - datetime.timedelta(days=7)
        str_7d = dt_7d.strftime('%Y-%m-%d')
        
        juegos = statsapi.schedule(team=team_id, start_date=str_7d, end_date=fecha_corte)
        if not juegos: return res
        
        total_outs = 0; total_hits = 0; total_bb = 0; total_k = 0; total_er = 0
        
        for juego in juegos:
            if juego.get('status') not in ['Final', 'Game Over']: continue
            game_id = juego.get('game_id')
            if not game_id: continue
            
            try:
                box = statsapi.boxscore_data(game_id)
                team_side = 'home' if juego.get('home_id') == team_id else 'away'
                pitchers_list = box.get(team_side, {}).get('pitchers', [])
                if len(pitchers_list) <= 1: continue # Solo lanzó el abridor
                
                relevistas = pitchers_list[1:] # Eliminamos al abridor
                for pid in relevistas:
                    p_key = f"ID{pid}"
                    p_stats = box.get(team_side, {}).get('players', {}).get(p_key, {}).get('stats', {}).get('pitching', {})
                    if not p_stats: continue
                        
                    total_hits += int(p_stats.get('hits', 0))
                    total_bb += int(p_stats.get('baseOnBalls', 0))
                    total_k += int(p_stats.get('strikeOuts', 0))
                    total_er += int(p_stats.get('earnedRuns', 0))
                    
                    ip_str = str(p_stats.get('inningsPitched', '0.0'))
                    if '.' in ip_str:
                        full, frac = ip_str.split('.')
                        total_outs += (int(full) * 3) + int(frac)
                    else:
                        total_outs += int(ip_str) * 3
            except: pass
                
        if total_outs > 0:
            ip_calc = total_outs / 3.0
            res["IP"] = f"{int(total_outs//3)}.{total_outs%3}"
            res["H"] = total_hits
            res["BB"] = total_bb
            res["K"] = total_k
            res["ER"] = total_er
            res["ERA"] = f"{(total_er * 9) / ip_calc:.2f}"
            res["WHIP"] = f"{(total_hits + total_bb) / ip_calc:.2f}"
    except Exception: pass
    return res
        
def convertir_df_a_excel(df, sheet_name="Hoja1"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    datos_procesados = output.getvalue()
    return datos_procesados

if 'df_mlb' not in st.session_state: st.session_state.df_mlb = None
if 'modelo_ia' not in st.session_state: st.session_state.modelo_ia = None
if 'fecha_modelo' not in st.session_state: st.session_state.fecha_modelo = None

st.sidebar.markdown("### 🗓️ Motor de Tiempo")
st.sidebar.markdown("Las fechas cambian estrictamente a las 12:00 AM Hora del Este (ET). Selecciona días pasados para auditar el rendimiento del radar.")

tz_et = 'America/New_York'
hoy_et = pd.Timestamp.now(tz_et).date()

fecha_sel = st.sidebar.date_input("Fecha de Análisis:", hoy_et)
st.session_state.fecha_hoy = fecha_sel.strftime('%Y-%m-%d')
st.sidebar.markdown("---")

st.sidebar.markdown("### 📥 Sincronización")
anio_sel = datetime.datetime.now().year

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
            st.session_state.modelo_ia = None
            st.sidebar.success("✅ Base de datos al día.")
        except Exception as e: st.sidebar.error(f"Error Crítico: {e}")

if st.session_state.df_mlb is not None:
    df_historico = st.session_state.df_mlb.copy()
    df_filtrado = df_historico[df_historico['Date'] < st.session_state.fecha_hoy].copy()
    
    # 🟢 INICIO DE LA OPTIMIZACIÓN SEGURA DE VELOCIDAD 🟢
    necesita_entrenar = True
    
    # Si la IA ya fue entrenada hoy, la sacamos de la memoria al instante
    if st.session_state.modelo_ia is not None and st.session_state.fecha_modelo == st.session_state.fecha_hoy:
        necesita_entrenar = False
        clf = st.session_state.modelo_ia
        
    if len(df_filtrado) > 0 and necesita_entrenar:
        st.sidebar.info("🧠 Entrenando IA. Tomará unos segundos la primera vez...")
        df_filtrado = df_filtrado.sort_values('Date').reset_index(drop=True)
        
        x_racha_diff, x_h2h, x_luck_diff, x_split_diff, y_win = [], [], [], [], []
        df_train = df_filtrado.tail(600).copy()
        
        for idx, row in df_train.iterrows():
            fecha_juego = row['Date']
            e_local = row['Local']
            e_visita = row['Visitante']
            
            df_pasado = df_filtrado[df_filtrado['Date'] < fecha_juego]
            
            if len(df_pasado) < 50:
                x_racha_diff.append(0.0)
                x_h2h.append(0.5)
                x_luck_diff.append(0.0)
                x_split_diff.append(0.0)
            else:
                r_l = get_recent_form(e_local, df_pasado)
                r_v = get_recent_form(e_visita, df_pasado)
                x_racha_diff.append(r_l - r_v)
                
                x_h2h.append(get_h2h_wins(e_local, e_visita, df_pasado))
                
                luck_l = get_pythagorean_luck(e_local, df_pasado)
                luck_v = get_pythagorean_luck(e_visita, df_pasado)
                x_luck_diff.append(luck_l - luck_v)
                
                s_l, s_v = get_splits_win_pct(e_local, e_visita, df_pasado)
                x_split_diff.append(s_l - s_v)
                
            y_win.append(1 if row['Carreras_Local'] > row['Carreras_Visitante'] else 0)
            
        df_train['Racha_Diff'] = x_racha_diff
        df_train['H2H_L_WinPct'] = x_h2h
        df_train['Luck_Diff'] = x_luck_diff
        df_train['Split_Diff'] = x_split_diff
        df_train['Win'] = y_win
        
        features = ['Elo_L', 'Elo_V', 'Racha_Diff', 'H2H_L_WinPct', 'Luck_Diff', 'Split_Diff']
        clf = RandomForestClassifier(n_estimators=150, max_depth=MAX_DEPTH_ELO, random_state=42)
        clf.fit(df_train[features], df_train['Win'])
        
        # Guardamos la IA en la caja fuerte para no repetir este ciclo en el próximo clic
        st.session_state.modelo_ia = clf
        st.session_state.fecha_modelo = st.session_state.fecha_hoy
        st.sidebar.success("✅ IA lista y guardada.")
    # 🟢 FIN DE LA OPTIMIZACIÓN 🟢
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📅 Cartelera del Día",
    "❄️ Caza-Bajas",
    "🔹 Caza-Hits",
    "🧮 Calculadora +EV",
    "📊 Auditoría Semanal",
    "🔬 Lupa de Pitcheo",
    "⭐ Filtros Premium"
])
    
    with tab1:
        st.markdown(f"### 🎯 Partidos programados para el: **{st.session_state.fecha_hoy}**")

        if st.button("⚡ Analizar y Evaluar Cartelera", type="primary", use_container_width=True):
            if len(df_filtrado) == 0:
                st.error("No hay suficientes datos históricos previos a esta fecha para entrenar el modelo.")
                st.session_state.resultados_jornada = None
            else:
                with st.spinner("Escaneando predicciones y verificando con resultados reales..."):
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
                            
                            home_id = juego.get('home_id')
                            away_id = juego.get('away_id')

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
                            
                            whip_bp_l = get_bullpen_metrics(home_id, st.session_state.fecha_hoy)
                            whip_bp_v = get_bullpen_metrics(away_id, st.session_state.fecha_hoy)

                            # 1. PREDICCIÓN MULTIVARIABLE (Enviamos las 6 variables a la IA)
                            X_hoy = np.array([[elo_l, elo_v, (racha_l - racha_v), h2h, (luck_l - luck_v), (split_l - split_v)]])
                            prob_ml = clf.predict_proba(X_hoy)[0][1]

                            # 2. AJUSTE DE PITCHEO MANUAL (Los lanzadores de hoy)
                            pitcher_adj = ((whip_v - whip_l) * 0.10) + ((whip_bp_v - whip_bp_l) * 0.05)
                            
                            prob_final_local = prob_ml + pitcher_adj

                            # Calculamos las variables del día de hoy
                            racha_diff = get_recent_form(e_local, df_filtrado) - get_recent_form(e_visita, df_filtrado)
                            h2h_l = get_h2h_wins(e_local, e_visita, df_filtrado)
                            luck_diff = get_pythagorean_luck(e_local, df_filtrado) - get_pythagorean_luck(e_visita, df_filtrado)
                            s_l, s_v = get_splits_win_pct(e_local, e_visita, df_filtrado)
                            split_diff = s_l - s_v

                            # Extraemos a los lanzadores
                            whip_l = get_pitcher_whip(p_local, st.session_state.fecha_hoy)
                            whip_v = get_pitcher_whip(p_visita, st.session_state.fecha_hoy)
                            whip_bp_l = get_bullpen_metrics(home_id, st.session_state.fecha_hoy)
                            whip_bp_v = get_bullpen_metrics(away_id, st.session_state.fecha_hoy)

                            # 1. PREDICCIÓN PURA DE LA IA (Machine Learning)
                            X_hoy = np.array([[elo_l, elo_v, racha_diff, h2h_l, luck_diff, split_diff]])
                            prob_ml = clf.predict_proba(X_hoy)[0][1]

                            # 2. AJUSTE DE PITCHEO MANUAL (Hybrid Model)
                            pitcher_adj = ((whip_v - whip_l) * 0.10) + ((whip_bp_v - whip_bp_l) * 0.05)
                            
                            prob_final_local = prob_ml + pitcher_adj

                            ganador = e_local if prob_final_local > 0.5 else e_visita
                            pct_bruto = prob_final_local if prob_final_local > 0.5 else 1.0 - prob_final_local
                            pct_final = int(round(max(min(pct_bruto, 0.99), 0.01) * 100))

                           # 🌟 FILTRO DE ALTA SEGURIDAD (GANADORES) 🌟
                            # 1. Probabilidad general altísima (>65%)
                            # 2. Mismatch de abridores: Nuestro abridor con WHIP élite (<1.15) vs Abridor descontrolado (>1.35)
                            # 3. Blindaje de Bullpen: Nuestro relevo NO puede ser un desastre (WHIP de Bullpen < 1.35)
                            es_alta_seg = False
                            if pct_final >= 65:
                                if ganador == e_local and whip_l < 1.15 and whip_v > 1.35 and whip_bp_l < 1.35:
                                    es_alta_seg = True
                                elif ganador == e_visita and whip_v < 1.15 and whip_l > 1.35 and whip_bp_v < 1.35:
                                    es_alta_seg = True

                            # Agregamos la estrella automática al nombre del equipo
                            jugada_str = f"⭐ {ganador} (A Ganar)" if es_alta_seg else f"{ganador} (A Ganar)"
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
                        st.success("✅ Análisis completado.")

        if "resultados_jornada" in st.session_state and st.session_state.resultados_jornada is not None:
            resultados_jornada = st.session_state.resultados_jornada
            df_resultados = pd.DataFrame(resultados_jornada).drop(columns=['score', 'raw_time'], errors='ignore')

            def color_whip(row):
                styles = [''] * len(row)
                for j, col in enumerate(row.index):
                    if col in ['⚾ Abridor (V)', '⚾ Abridor (L)']:
                        try:
                            val_str = str(row[col])
                            whip = float(val_str.split('(')[-1].replace(')', '')) if '(' in val_str else float(val_str)
                            
                            if whip < 1.00: styles[j] = 'color: #00cc66; font-weight: bold;'
                            elif whip <= 1.30: styles[j] = 'color: #e6b800; font-weight: bold;'
                            else: styles[j] = 'color: #ff4d4d; font-weight: bold;'
                        except: pass
                return styles

            df_estilizado = df_resultados.style.apply(color_whip, axis=1)\
                .set_properties(**{'text-align': 'center'})\
                .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

            st.dataframe(df_estilizado, use_container_width=True, hide_index=True)
            
            # --- NUEVA NOTA ANALÍTICA ---
            st.caption("📝 **Nota Analítica:** El dato junto a los abridores muestra el WHIP de sus últimas 7 salidas.")

            excel_cartelera = convertir_df_a_excel(df_resultados, "Cartelera")
            st.download_button(
                label="📥 Descargar Cartelera (Excel)",
                data=excel_cartelera,
                file_name=f"cartelera_mlb_{st.session_state.fecha_hoy}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

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
        st.markdown("### ❄️ Radar de Bajas: Pitcher K/9 vs Vulnerabilidad del Rival")
        if st.button("🎯 Cazar Bajas del Día (Top 4)", type="primary", use_container_width=True):
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

            excel_ponches = convertir_df_a_excel(df_k, "Ponches")
            st.download_button(
                label="📥 Descargar Caza-Bajas (Excel)",
                data=excel_ponches,
                file_name=f"caza_ponches_{st.session_state.fecha_hoy}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            total_evaluados = sum(1 for e in df_k['📝 Evaluación'] if '✅' in e or '❌' in e)
            aciertos = sum(1 for e in df_k['📝 Evaluación'] if '✅' in e)

            if total_evaluados > 0:
                efectividad = (aciertos / total_evaluados) * 100
                st.markdown("### 📊 Rendimiento Caza-Bajas")
                c1, c2, c3 = st.columns(3)
                c1.metric("Lanzadores Evaluados", total_evaluados)
                c2.metric("Metas Superadas", aciertos)
                c3.metric("Efectividad", f"{int(round(efectividad))}%")
        elif clave_k not in st.session_state:
            st.info("Presiona el botón para cazar bajas del día.")
                
    with tab3:
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

            excel_hits = convertir_df_a_excel(df_hits, "Hits")
            st.download_button(
                label="📥 Descargar Caza-Hits (Excel)",
                data=excel_hits,
                file_name=f"caza_hits_{st.session_state.fecha_hoy}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            total_evaluados = sum(1 for e in df_hits['📝 Evaluación'] if '✅' in e or '❌' in e)
            aciertos = sum(1 for e in df_hits['📝 Evaluación'] if '✅' in e)

            if total_evaluados > 0:
                efectividad = (aciertos / total_evaluados) * 100
                st.markdown("### 📊 Rendimiento Caza-Hits (1+ Imparables)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Bateadores Evaluados", total_evaluados)
                c2.metric("Aciertos (1+ Hits)", aciertos)
                c3.metric("Efectividad", f"{int(round(efectividad))}%")
        elif clave_hits not in st.session_state:
            st.info("Presiona el botón para buscar bateadores con alta probabilidad de 1+ hits.")

    with tab4:
        st.markdown("### 🧮 Calculadora de Valor Esperado (+EV)")
        st.markdown("Compara la probabilidad matemática del Radar con la cuota decimal de tu casa de apuestas para descubrir si la jugada es rentable a largo plazo.")
        
        c1, c2 = st.columns(2)
        with c1:
            prob_radar = st.number_input("📊 Probabilidad que arrojó el Radar (%)", min_value=1, max_value=99, value=55, step=1)
        with c2:
            cuota_decimal = st.number_input("🏦 Cuota Decimal (ej. 1.91, 2.50)", min_value=1.01, value=1.91, step=0.01)
            
        if cuota_decimal > 1.0:
            prob_implicita = 1.0 / cuota_decimal
            
            prob_radar_dec = prob_radar / 100.0
            ev_pct = (prob_radar_dec * cuota_decimal) - 1.0
            
            prob_implicita_int = int(round(prob_implicita * 100))
            ev_pct_int = int(round(ev_pct * 100))
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Probabilidad que exige el Casino", f"{prob_implicita_int}%")
            
            if ev_pct >= 0.10:
                st.toast('Apuesta de alto valor (+10% EV)', icon='💎')
                st.success("💎 Apuesta de alto valor (+10% EV)")
            elif ev_pct >= 0.05:
                st.success("✅ Apuesta recomendada (+5% EV)")
            elif ev_pct > 0:
                st.warning("⚠️ Valor marginal, solo si el radar es muy fiable")
            else:
                st.error("❌ Apuesta no rentable")
            
            if ev_pct > 0:
                col2.metric("Valor Esperado (EV)", f"+{ev_pct_int}%", "Rentable (+EV)")
                st.success(f"✅ **¡Apuesta de Valor!** El radar le da **{prob_radar}%** de probabilidad de éxito, y la casa de apuestas te está cobrando como si solo tuviera **{prob_implicita_int}%**. Tienes ventaja matemática. Si repites esta apuesta 100 veces, ganarás dinero.")
            else:
                col2.metric("Valor Esperado (EV)", f"{ev_pct_int}%", "No Rentable (-EV)", delta_color="inverse")
                st.error(f"❌ **Déjala Pasar.** El casino está protegiendo su dinero exigiendo un **{prob_implicita_int}%** de éxito, pero el radar solo le da un **{prob_radar}%**. A largo plazo, esta apuesta te hará perder tu capital (bankroll).")

    with tab5:
        st.markdown("### 📊 Auditoría Premium (Últimos 7 Días)")
        st.markdown("Evalúa estrictamente la rentabilidad de las jugadas de Alta Seguridad (⭐). Al excluir el escaneo masivo de hits, esta auditoría carga mucho más rápido.")

        if st.button("🔍 Ejecutar Auditoría Premium", type="primary", use_container_width=True):
            hoy = datetime.date.today()
            fechas_auditar = [(hoy - datetime.timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]

            resultados = []
            barra_progreso = st.progress(0)
            estado = st.empty()
            
            # Llamamos al cerebro principal que ya está entrenado con las 6 variables
            clf_principal = st.session_state.modelo_ia 

            for idx, fecha_str in enumerate(fechas_auditar):
                estado.write(f"⏳ Procesando {fecha_str} (Analizando pitcheo de élite)...")

                juegos_dia = statsapi.schedule(date=fecha_str, sportId=1)
                if not any(j['status'] in ['Final', 'Game Over'] for j in juegos_dia):
                    barra_progreso.progress((idx + 1) / len(fechas_auditar))
                    continue

                df_filtrado_aud = df_historico[df_historico['Date'] < fecha_str].copy()
                if len(df_filtrado_aud) == 0:
                    barra_progreso.progress((idx + 1) / len(fechas_auditar))
                    continue

                aciertos_gan_premium = 0
                total_gan_premium = 0
                
                for juego in juegos_dia:
                    if juego['status'] not in ['Final', 'Game Over']: continue
                    e_local = juego['home_name']
                    e_visita = juego['away_name']
                    if e_local not in MLB_TEAM_WHITELIST or e_visita not in MLB_TEAM_WHITELIST: continue
                    
                    home_id = juego.get('home_id')
                    away_id = juego.get('away_id')

                    p_local, p_visita = get_starting_pitchers(juego)
                    elo_l = df_filtrado_aud[df_filtrado_aud['Local'] == e_local].tail(1)['Elo_L'].values[0] if len(df_filtrado_aud[df_filtrado_aud['Local'] == e_local]) > 0 else 1500
                    elo_v = df_filtrado_aud[df_filtrado_aud['Visitante'] == e_visita].tail(1)['Elo_V'].values[0] if len(df_filtrado_aud[df_filtrado_aud['Visitante'] == e_visita]) > 0 else 1500
                    elo_l += 35

                    racha_l = get_recent_form(e_local, df_filtrado_aud)
                    racha_v = get_recent_form(e_visita, df_filtrado_aud)
                    h2h = get_h2h_wins(e_local, e_visita, df_filtrado_aud)
                    luck_l = get_pythagorean_luck(e_local, df_filtrado_aud)
                    luck_v = get_pythagorean_luck(e_visita, df_filtrado_aud)
                    split_l, split_v = get_splits_win_pct(e_local, e_visita, df_filtrado_aud)
                    
                    whip_l = get_pitcher_whip(p_local, fecha_str)
                    whip_v = get_pitcher_whip(p_visita, fecha_str)
                    
                    whip_bp_l = get_bullpen_metrics(home_id, fecha_str)
                    whip_bp_v = get_bullpen_metrics(away_id, fecha_str)

                    # Usamos el cerebro IA principal con las 6 variables intactas
                    X_auditoria = np.array([[elo_l, elo_v, (racha_l - racha_v), h2h, (luck_l - luck_v), (split_l - split_v)]])
                    prob_ml = clf_principal.predict_proba(X_auditoria)[0][1]
                    
                    pitcher_adj = ((whip_v - whip_l) * 0.10) + ((whip_bp_v - whip_bp_l) * 0.05)
                    prob_final_local = prob_ml + pitcher_adj
                    
                    ganador = e_local if prob_final_local > 0.5 else e_visita
                    pct_bruto = prob_final_local if prob_final_local > 0.5 else 1.0 - prob_final_local
                    pct_final = int(round(max(min(pct_bruto, 0.99), 0.01) * 100))

                    # 🌟 FILTRO DE LA ESTRELLA PARA LA AUDITORÍA
                    es_alta_seg = False
                    if pct_final >= 68:
                        if ganador == e_local and whip_l < 1.15 and whip_v > 1.35:
                            es_alta_seg = True
                        elif ganador == e_visita and whip_v < 1.15 and whip_l > 1.35:
                            es_alta_seg = True

                    # Si cumplió las reglas Premium, la evaluamos
                    if es_alta_seg:
                        r_local = juego.get('home_score', 0)
                        r_visita = juego.get('away_score', 0)
                        r_ganador = e_local if r_local > r_visita else e_visita
                        if r_ganador == ganador: aciertos_gan_premium += 1
                        total_gan_premium += 1

                # 🌟 EVALUACIÓN ESTRICTA DE CAZA-BAJAS PREMIUM
                k_data = get_strikeout_hunters(fecha_str)
                # Filtramos la lista para auditar solo los que se ganaron la estrella
                premium_k_data = [k for k in k_data if '⭐' in k['⚾ Abridor']]
                
                aciertos_k = sum(1 for k in premium_k_data if '✅' in k['📝 Evaluación'])
                fallos_k = sum(1 for k in premium_k_data if '❌' in k['📝 Evaluación'])
                total_k = aciertos_k + fallos_k

                resultados.append({
                    "Fecha": fecha_str,
                    "Ganadores ⭐": f"{aciertos_gan_premium}/{total_gan_premium}",
                    "Bajas (Under) ⭐": f"{aciertos_k}/{total_k}",
                    "Efect. Ganadores (%)": round(aciertos_gan_premium/total_gan_premium*100, 1) if total_gan_premium else 0,
                    "Efect. Bajas (%)": round(aciertos_k/total_k*100, 1) if total_k else 0
                })

                barra_progreso.progress((idx + 1) / len(fechas_auditar))
                time.sleep(0.2)

            estado.empty()
            barra_progreso.empty()

            st.session_state.auditoria_7dias = resultados

        if "auditoria_7dias" in st.session_state and st.session_state.auditoria_7dias:
            resultados = st.session_state.auditoria_7dias
            if resultados:
                df_aud = pd.DataFrame(resultados)
                st.markdown("### 📈 Resultados Diarios (Solo Jugadas ⭐)")
                st.dataframe(df_aud, use_container_width=True, hide_index=True)
                
                excel_auditoria = convertir_df_a_excel(df_aud, "Auditoria")
                st.download_button(
                    label="📥 Descargar Auditoría (Excel)",
                    data=excel_auditoria,
                    file_name="auditoria_premium_7dias.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                total_gan_acc = sum(int(r["Ganadores ⭐"].split('/')[0]) for r in resultados)
                total_gan_eval = sum(int(r["Ganadores ⭐"].split('/')[1]) for r in resultados)
                total_k_acc = sum(int(r["Bajas (Under) ⭐"].split('/')[0]) for r in resultados)
                total_k_eval = sum(int(r["Bajas (Under) ⭐"].split('/')[1]) for r in resultados)

                st.markdown("---")
                st.markdown("### 📊 Resumen Acumulado de Élite (7 días)")
                col1, col2 = st.columns(2)
                col1.metric("Aciertos Ganadores Premium", f"{total_gan_acc}/{total_gan_eval}", f"{round(total_gan_acc/total_gan_eval*100,1)}%" if total_gan_eval else "0%")
                col2.metric("Aciertos Bajas Premium", f"{total_k_acc}/{total_k_eval}", f"{round(total_k_acc/total_k_eval*100,1)}%" if total_k_eval else "0%")
            else:
                st.warning("No se encontraron juegos finalizados en los últimos 7 días.")
                
    with tab6:
        st.markdown("### 🔬 Lupa de Pitcheo: Radiografía de 7 Juegos")
        st.markdown("Selecciona un partido de la cartelera para desglosar el desempeño crudo del Abridor y el Bullpen en su muestra reciente.")
        
        juegos_dia_lupa = statsapi.schedule(date=st.session_state.fecha_hoy, sportId=1)
        juegos_validos = [j for j in juegos_dia_lupa if j.get('status') not in ['Postponed', 'Cancelled']]
        
        if not juegos_validos:
            st.info("No hay juegos programados para esta fecha.")
        else:
            opciones_juegos = {f"{j['away_name']} ✈️ @ 🏠 {j['home_name']}": j for j in juegos_validos}
            juego_sel = st.selectbox("⚾ Selecciona el Partido:", list(opciones_juegos.keys()))
            
            if st.button("🔍 Extraer Radiografía", type="primary", use_container_width=True):
                with st.spinner("Extrayendo estadísticas quirúrgicas de los Boxscores..."):
                    j_data = opciones_juegos[juego_sel]
                    
                    p_home, p_away = get_starting_pitchers(j_data)
                    home_id = j_data['home_id']
                    away_id = j_data['away_id']
                    home_name = j_data['home_name']
                    away_name = j_data['away_name']
                    
                    # Away
                    away_starter_stats = get_detailed_pitcher_stats(p_away, st.session_state.fecha_hoy)
                    away_bullpen_stats = get_detailed_bullpen_stats(away_id, st.session_state.fecha_hoy)
                    
                    # Home
                    home_starter_stats = get_detailed_pitcher_stats(p_home, st.session_state.fecha_hoy)
                    home_bullpen_stats = get_detailed_bullpen_stats(home_id, st.session_state.fecha_hoy)
                    
                    datos_lupa = [
                        {"Equipo": away_name, "Rol": f"Abridor: {p_away or 'TBD'}", **away_starter_stats},
                        {"Equipo": away_name, "Rol": "Bullpen (Relevistas)", **away_bullpen_stats},
                        {"Equipo": home_name, "Rol": f"Abridor: {p_home or 'TBD'}", **home_starter_stats},
                        {"Equipo": home_name, "Rol": "Bullpen (Relevistas)", **home_bullpen_stats},
                    ]
                    
                    df_lupa = pd.DataFrame(datos_lupa)
                    
                    def color_lupa_whip(val):
                        try:
                            v = float(val)
                            if v < 1.00: return 'color: #00cc66; font-weight: bold;' # Verde (< 1.00)
                            elif v <= 1.30: return 'color: #e6b800; font-weight: bold;' # Amarillo (1.00 a 1.30)
                            else: return 'color: #ff4d4d; font-weight: bold;' # Rojo (> 1.30)
                        except: return ''
                        
                    def color_lupa_era(val):
                        try:
                            v = float(val)
                            if v < 3.20: return 'color: #00cc66; font-weight: bold;'
                            elif v <= 4.20: return 'color: #ff9900; font-weight: bold;'
                            else: return 'color: #ff4d4d; font-weight: bold;'
                        except: return ''
                    
                    df_lupa_estilizado = df_lupa.style\
                        .map(color_lupa_whip, subset=['WHIP'])\
                        .map(color_lupa_era, subset=['ERA'])\
                        .set_properties(**{'text-align': 'center'})\
                        .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
                    
                    st.dataframe(df_lupa_estilizado, use_container_width=True, hide_index=True)
                    
                    st.caption("📝 **Nota Analítica:** La tabla muestra las últimas 7 salidas de los abridores y todo el trabajo de los relevistas en los últimos 7 días. ER = Carreras Limpias.")

    with tab7:
        st.markdown("### ⭐ Manual de Jugadas Premium")
        st.markdown("El radar asigna automáticamente una estrella (⭐) a las jugadas que cumplen con estrictos criterios matemáticos de altísima seguridad. Aquí están las reglas exactas que el algoritmo exige de forma interna para encender la alerta de **Jugada Premium**.")
        
        st.markdown("---")
        
        st.markdown("#### 🎯 Cartelera del Día (A Ganar)")
        st.markdown("Para que un equipo reciba la estrella de seguridad, debe superar una evaluación obligatoria en 3 dimensiones:")
        st.markdown("* **🧠 Probabilidad IA (>=65%):** El modelo multivariable (Elo, Rachas, Suerte Pitagórica, Splits) debe otorgarle al menos un 65% de probabilidad base de ganar el encuentro.")
        st.markdown("* **⚔️ Mismatch de Abridores:** Nuestro abridor debe ser la élite reciente (WHIP menor a **1.15**), mientras que el abridor rival debe mostrar grave descontrol o ser bateado con facilidad (WHIP mayor a **1.35**).")
        st.markdown("* **🛡️ Blindaje de Bullpen:** El relevo de nuestro equipo no puede arruinar la ventaja. Exigimos que el bullpen haya mantenido un WHIP por debajo de **1.35** en los últimos 7 días.")

        st.markdown("---")

        st.markdown("#### ❄️ Caza-Bajas (Under 4.5 Ponches)")
        st.markdown("Para que la Baja de un lanzador sea considerada una jugada maestra, debe cumplir estas 3 reglas inquebrantables simultáneamente:")
        st.markdown("* **📈 Probabilidad Extrema (>=85%):** La fórmula de Poisson debe proyectar al menos un 85% de posibilidades reales de que el lanzador logre 4 ponches o menos.")
        st.markdown("* **⏱️ Correa Corta (Proy. IP <= 4.0):** El mánager no debe dejarlo pasar del quinto inning. Matemáticamente, a menos outs lanzados, menos oportunidades de sumar ponches fortuitos.")
        st.markdown("* **🧊 Lanzador de Contacto (K/9 <= 6):** El lanzador debe tener una tendencia natural a inducir batazos de out en el cuadro en lugar de abanicar bateadores (Promedio de 6 ponches o menos cada 9 innings).")

        st.markdown("---")

        st.markdown("#### 🔹 Caza-Hits (1+ Imparables)")
        st.markdown("El mercado de hits es de alta varianza. Para que un bateador obtenga la estrella, debe enfrentar la 'Tormenta Perfecta' en el plato:")
        st.markdown("* **🎯 Probabilidad Base (>=80%):** El modelo binomial debe calcular un 80% o más de probabilidad de éxito.")
        st.markdown("* **🔥 Bateador Encendido (AVG L10 >= .300):** El jugador debe estar viendo la pelota a la perfección, bateando para .300 o más en sus últimos 10 juegos.")
        st.markdown("* **🛡️ Pitcheo Global Vulnerable (WHIP >= 1.30):** No basta con un mal abridor; el WHIP combinado (60% Abridor + 40% Bullpen) debe ser de 1.30 o superior, garantizando que el bateador enfrentará lanzadores permisivos durante los 9 innings.")
        st.markdown("* **⚔️ Ventaja de Pelotón (Platoon Advantage):** El bateador debe pararse en el plato del lado opuesto al brazo de lanzar del abridor (Ej: Bateador Zurdo vs Pitcher Derecho), obteniendo la máxima ventaja visual.")
