import streamlit as st
import pandas as pd
import numpy as np
import statsapi
import time
import datetime
import calendar
from sklearn.ensemble import RandomForestClassifier

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

def get_pitcher_whip(pitcher_name):
    avg_whip = 1.30 
    if not pitcher_name or pitcher_name == 'TBD': 
        return avg_whip
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
                                splits.sort(key=lambda x: x.get('date', ''), reverse=True)
                                last_7 = splits[:7]
                                
                                total_hits = 0
                                total_bb = 0
                                total_outs = 0
                                
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
                                        
                                if total_outs > 0:
                                    whip = (total_hits + total_bb) / (total_outs / 3.0)
                                    return round(whip, 2)
                                else:
                                    return avg_whip
        except Exception:
            pass
            
        try:
            stats_season = statsapi.player_stat_data(player_id, group="pitching", type="season")
            if stats_season and 'stats' in stats_season and len(stats_season['stats']) > 0:
                whip_str = stats_season['stats'][0]['stats'].get('whip', '-.--')
                if whip_str != '-.--':
                    return float(whip_str)
        except Exception:
            pass
                
        return avg_whip
    except:
        return avg_whip

def get_hr_hunters(anio, fecha_hoy):
    try:
        juegos_hoy = statsapi.schedule(date=fecha_hoy, sportId=1)
        equipos_hoy = {}
        for juego in juegos_hoy:
            if juego.get('status', '') not in ['Postponed', 'Cancelled']:
                equipos_hoy[juego.get('home_id')] = 'Local'
                equipos_hoy[juego.get('away_id')] = 'Visitante'
        
        data = statsapi.get('stats_leaders', {'leaderCategories': 'homeRuns', 'season': anio, 'limit': 60, 'statGroup': 'hitting'})
        if not data or 'leagueLeaders' not in data or len(data['leagueLeaders']) == 0:
            return []
            
        leaders = data['leagueLeaders'][0].get('leaders', [])
        
        jugadores_activos = []
        for p in leaders:
            team_id = p.get('team', {}).get('id')
            team_name = p.get('team', {}).get('name', 'Unknown')
            if team_id in equipos_hoy:
                p['condicion_hoy'] = equipos_hoy[team_id]
                p['team_name'] = team_name
                jugadores_activos.append(p)
                
        resultados = []
        ayer_dt = datetime.datetime.strptime(fecha_hoy, '%Y-%m-%d') - datetime.timedelta(days=1)
        fecha_ayer_str = ayer_dt.strftime('%Y-%m-%d')
        
        for p in jugadores_activos:
            p_id = p.get('person', {}).get('id')
            p_name = p.get('person', {}).get('fullName')
            team_name = p.get('team_name', 'Unknown')
            season_hr = int(p.get('value', 0))
            condicion = p.get('condicion_hoy', 'Visitante')
            
            raw_data = statsapi.get('people', {'personIds': p_id, 'hydrate': 'stats(group=[hitting],type=[season,gameLog])'})
            person = raw_data.get('people', [{}])[0]
            stats_blocks = person.get('stats', [])
            
            season_ab = 1
            l10_hr = 0
            l10_ab = 0
            dio_jonron_ayer = False
            
            for block in stats_blocks:
                if block.get('type', {}).get('displayName') == 'season':
                    season_ab = max(1, int(block.get('splits', [{}])[0].get('stat', {}).get('atBats', 1)))
                elif block.get('type', {}).get('displayName') == 'gameLog':
                    splits = block.get('splits', [])
                    splits.sort(key=lambda x: x.get('date', ''), reverse=True)
                    last_10 = splits[:10]
                    
                    for game in last_10:
                        g_date = game.get('date', '')
                        g_stats = game.get('stat', {})
                        hr_en_juego = int(g_stats.get('homeRuns', 0))
                        
                        if g_date == fecha_ayer_str and hr_en_juego > 0:
                            dio_jonron_ayer = True
                            
                        l10_hr += hr_en_juego
                        l10_ab += int(g_stats.get('atBats', 0))
            
            if dio_jonron_ayer:
                continue
                
            l10_ab = max(1, l10_ab)
            freq_season = season_hr / season_ab
            freq_recent = l10_hr / l10_ab
            
            bono_localia = 1.10 if condicion == 'Local' else 1.0 
            hr_index = ((freq_season * 0.3) + (freq_recent * 0.7)) * bono_localia
            
            resultados.append({
                "⚾ Bateador": p_name,
                "👕 Equipo": team_name,
                "🏟️ Condición": condicion,
                "🏆 HR Año": season_hr,
                "🔥 HR (L10)": l10_hr,
                "📊 Turnos (L10)": l10_ab,
                "score": hr_index
            })
            
        resultados.sort(key=lambda x: x['score'], reverse=True)
        resultados = resultados[:4]
        
        tabla_final = []
        for r in resultados:
            tabla_final.append({
                "⚾ Bateador": r["⚾ Bateador"],
                "👕 Equipo": r["👕 Equipo"],
                "🏟️ Condición": r["🏟️ Condición"],
                "🏆 HR Año": r["🏆 HR Año"],
                "🔥 HR (L10)": r["🔥 HR (L10)"],
                "📈 Ratio de Poder (L10)": f"{r['🔥 HR (L10)']} HR / {r['📊 Turnos (L10)']} VB"
            })
            
        return tabla_final
    except Exception as e:
        return []

def get_strikeout_hunters(fecha_hoy):
    try:
        juegos_hoy = statsapi.schedule(date=fecha_hoy, sportId=1)
        if not juegos_hoy: return []
        
        pitchers_data = []
        
        for juego in juegos_hoy:
            if juego.get('status', '') in ['Postponed', 'Cancelled']: continue
            
            matchups = [
                (juego.get('home_probable_pitcher'), juego.get('home_name'), juego.get('away_id'), juego.get('away_name')),
                (juego.get('away_probable_pitcher'), juego.get('away_name'), juego.get('home_id'), juego.get('home_name'))
            ]
            
            for p_name, p_team, opp_id, opp_name in matchups:
                if not p_name or p_name == 'TBD': continue
                
                players = statsapi.lookup_player(p_name)
                if not players: continue
                p_id = players[0]['id']
                
                raw_data = statsapi.get('people', {'personIds': p_id, 'hydrate': 'stats(group=[pitching],type=[gameLog])'})
                person = raw_data.get('people', [{}])[0]
                stats_blocks = person.get('stats', [])
                
                l7_ks = 0
                l7_outs = 0
                juegos_lanzados = 0
                
                for block in stats_blocks:
                    if block.get('type', {}).get('displayName') == 'gameLog':
                        splits = block.get('splits', [])
                        splits.sort(key=lambda x: x.get('date', ''), reverse=True)
                        last_7 = splits[:7]
                        juegos_lanzados = len(last_7)
                        
                        for game in last_7:
                            g_stats = game.get('stat', {})
                            l7_ks += int(g_stats.get('strikeOuts', 0))
                            ip_str = str(g_stats.get('inningsPitched', '0.0'))
                            if '.' in ip_str:
                                full, frac = ip_str.split('.')
                                l7_outs += (int(full) * 3) + int(frac)
                            else:
                                l7_outs += int(ip_str) * 3
                                
                if juegos_lanzados == 0 or l7_outs == 0: continue
                
                ip_per_start = (l7_outs / 3.0) / juegos_lanzados
                k_per_9 = (l7_ks / (l7_outs / 3.0)) * 9.0
                avg_k_per_start = l7_ks / juegos_lanzados
                
                team_raw = statsapi.get('teams', {'teamId': opp_id, 'hydrate': 'stats(group=[hitting],type=[season])'})
                opp_ks = 0
                opp_pa = 1
                try:
                    t_stats = team_raw['teams'][0]['teamStats'][0]['splits'][0]['stat']
                    opp_ks = int(t_stats.get('strikeOuts', 0))
                    opp_pa = int(t_stats.get('plateAppearances', 1))
                except: pass
                    
                opp_k_pct = opp_ks / opp_pa if opp_pa > 1 else 0.225
                
                k_modifier = opp_k_pct / 0.225
                proj_k = avg_k_per_start * k_modifier
                
                pitchers_data.append({
                    "⚾ Abridor": p_name,
                    "👕 Equipo": p_team,
                    "⚔️ Rival": opp_name,
                    "🔥 K/9 (L7)": int(round(k_per_9)),
                    "📉 K% Rival": f"{opp_k_pct*100:.1f}%",
                    "🎯 Proy. Ponches": int(round(proj_k)), 
                    "score": proj_k
                })
                
        pitchers_data.sort(key=lambda x: x['score'], reverse=True)
        top_4 = pitchers_data[:4]
        
        for r in top_4:
            r["🎯 Proy. Ponches"] = f"{r['🎯 Proy. Ponches']} Ks" 
            del r['score']
            
        return top_4
        
    except Exception as e:
        return []

# --- ESTADO DE SESIÓN ---
if 'df_mlb' not in st.session_state: st.session_state.df_mlb = None
if 'fecha_hoy' not in st.session_state: st.session_state.fecha_hoy = datetime.date.today().strftime('%Y-%m-%d')
if 'resultados_hoy' not in st.session_state: st.session_state.resultados_hoy = None
if 'resultados_hr' not in st.session_state: st.session_state.resultados_hr = None
if 'resultados_k' not in st.session_state: st.session_state.resultados_k = None

# --- BARRA LATERAL: MOTOR DE DATOS ---
st.sidebar.markdown("### 📥 Sincronización")
anio_sel = st.sidebar.selectbox("Temporada a procesar:", [2026, 2025, 2024])

if st.sidebar.button("🔄 Descargar Historial Base", type="primary"):
    with st.spinner("Actualizando base de datos central... (Esto tomará un minuto)"):
        try:
            fechas = []
            for m in range(3, 12):
                last_day = calendar.monthrange(anio_sel, m)[1]
                fechas.append((f"{anio_sel}-{m:02d}-01", f"{anio_sel}-{m:02d}-{last_day}"))
                
            data_total = []
            for start, end in fechas:
                intentos = 0
                exito = False
                while intentos < 3 and not exito:
                    try:
                        batch = statsapi.schedule(start_date=start, end_date=end, sportId=1)
                        if batch: data_total.extend(batch)
                        exito = True
                        time.sleep(1.0)
                    except Exception:
                        intentos += 1
                        time.sleep(2.0)
                
                if not exito:
                    st.sidebar.warning(f"⚠️ El servidor de la MLB omitió el bloque de {start}.")
            
            df = pd.DataFrame(data_total)
            df = df[df['status'] == 'Final'].copy()
            
            if 'game_type' in df.columns:
                df = df[df['game_type'].isin(['R', 'P'])]
                
            df = df[['home_name', 'away_name', 'home_score', 'away_score', 'game_date']]
            df.columns = ['Local', 'Visitante', 'Carreras_Local', 'Carreras_Visitante', 'Date']
            df = df[df['Local'].isin(MLB_TEAM_WHITELIST) & df['Visitante'].isin(MLB_TEAM_WHITELIST)]
            
            elo_dict = {team: 1500.0 for team in MLB_TEAM_WHITELIST}
            h_elo_l, h_elo_v = [], []
            for _, row in df.iterrows():
                l, v = row['Local'], row['Visitante']
                el, ev = elo_dict[l], elo_dict[v]
                h_elo_l.append(el); h_elo_v.append(ev)
                diff = 1 / (1 + 10 ** ((ev - el) / 400))
                res = 1.0 if row['Carreras_Local'] > row['Carreras_Visitante'] else 0.0
                elo_dict[l] += 6 * (res - diff)
                elo_dict[v] += 6 * ((1 - res) - (1 - diff))
            
            df['Elo_L'], df['Elo_V'] = h_elo_l, h_elo_v
            st.session_state.df_mlb = df
            st.sidebar.success("✅ Base de datos al día. Juegos de exhibición purgados")
        except Exception as e: st.sidebar.error(f"Error Crítico: {e}")

# --- ÁREA PRINCIPAL ---
if st.session_state.df_mlb is not None:
    df = st.session_state.df_mlb.copy()
    
    df['Win'] = (df['Carreras_Local'] > df['Carreras_Visitante']).astype(int)
    clf = RandomForestClassifier(max_depth=MAX_DEPTH_ELO, random_state=42).fit(df[['Elo_L', 'Elo_V']], df['Win'])
    
    tab1, tab2, tab3 = st.tabs(["📅 Cartelera del Día", "💣 Caza-Jonrones", "🔥 Caza-Ponches (K-Props)"])
    
    with tab1:
        st.markdown(f"### 🎯 Partidos programados para hoy: **{st.session_state.fecha_hoy}**")
        st.markdown("El radar escanea la jornada completa, determina la **única mejor jugada** por partido (Ganador vs Totales)")
        
        if st.button("⚡ Analizar y Extraer Mejores Jugadas", type="primary", use_container_width=True):
            with st.spinner("Procesando todos los juegos, evaluando WHIP de abridores y rankeando las mejores oportunidades..."):
                juegos_hoy = statsapi.schedule(date=st.session_state.fecha_hoy, sportId=1)
                
                if not juegos_hoy:
                    st.warning("No hay juegos oficiales programados para la fecha de hoy.")
                    st.session_state.resultados_hoy = None
                else:
                    resultados_jornada = []
                    equipos_procesados = set() 
                    
                    for juego in juegos_hoy:
                        estado_juego = juego.get('status', '')
                        estados_validos = ['Scheduled', 'Pre-Game', 'Warmup', 'Delayed Start']
                        if estado_juego not in estados_validos:
                            continue
                            
                        e_local = juego['home_name']
                        e_visita = juego['away_name']
                        p_local = juego.get('home_probable_pitcher', '')
                        p_visita = juego.get('away_probable_pitcher', '')
                        
                        if e_local not in MLB_TEAM_WHITELIST or e_visita not in MLB_TEAM_WHITELIST: continue
                        if e_local in equipos_procesados or e_visita in equipos_procesados: continue
                        
                        equipos_procesados.add(e_local)
                        equipos_procesados.add(e_visita)
                            
                        game_dt_str = juego.get('game_datetime', '')
                        if game_dt_str:
                            try:
                                dt = pd.to_datetime(game_dt_str)
                                if dt.tzinfo is None:
                                    dt = dt.tz_localize('UTC')
                                hora_et = dt.tz_convert('America/New_York').strftime('%I:%M %p')
                            except:
                                hora_et = 'TBD'
                        else: 
                            hora_et = 'TBD'
                            
                        rec_l = get_team_record(e_local, df)
                        rec_v = get_team_record(e_visita, df)
                        
                        elo_l = df[df['Local'] == e_local].tail(1)['Elo_L'].values[0] if len(df[df['Local'] == e_local]) > 0 else 1500
                        elo_v = df[df['Visitante'] == e_visita].tail(1)['Elo_V'].values[0] if len(df[df['Visitante'] == e_visita]) > 0 else 1500
                        elo_l += 35 
                        
                        racha_l = get_recent_form(e_local, df)
                        racha_v = get_recent_form(e_visita, df)
                        h2h = get_h2h_wins(e_local, e_visita, df)
                        luck_l = get_pythagorean_luck(e_local, df)
                        luck_v = get_pythagorean_luck(e_visita, df)
                        split_l, split_v = get_splits_win_pct(e_local, e_visita, df)
                        
                        whip_l = get_pitcher_whip(p_local)
                        whip_v = get_pitcher_whip(p_visita)
                        
                        prob = clf.predict_proba(np.array([[elo_l, elo_v]]))[0][1]
                        pitcher_adj = (whip_v - whip_l) * 0.15 
                        
                        prob_final_local = (prob + 
                                            (racha_l - racha_v) * PESO_RACHA + 
                                            (h2h - 0.5) * PESO_H2H + 
                                            (luck_l - luck_v) * PESO_PITAGORICO + 
                                            (split_l - split_v) * PESO_SPLITS +
                                            pitcher_adj)
                        
                        if prob_final_local > 0.5:
                            ganador = e_local
                            pct_bruto = prob_final_local
                        else:
                            ganador = e_visita
                            pct_bruto = 1.0 - prob_final_local
                            
                        pct_final = int(round(max(min(pct_bruto, 0.99), 0.01) * 100))
                        
                        c_v, c_l = get_hybrid_run_projection(e_visita, e_local, df)
                        adj_runs_l = (whip_v - 1.30) * 1.5
                        adj_runs_v = (whip_l - 1.30) * 1.5
                        
                        c_l = max(0.5, c_l + adj_runs_l)
                        c_v = max(0.5, c_v + adj_runs_v)
                        total_runs = round(c_v + c_l, 2)
                        
                        if total_runs > LINEA_TOTALES: ou_pick = "ALTA"
                        else: ou_pick = "BAJA"
                        
                        total_runs_int = int(round(total_runs))
                        diff_total = abs(total_runs - LINEA_TOTALES)
                        pseudo_prob = int(round(50 + (diff_total * 10)))
                        
                        if pct_final >= pseudo_prob:
                            jugada_str = f"{ganador} (A Ganar)"
                            prob_str = f"{pct_final}%"
                            score_val = pct_final
                        else:
                            jugada_str = f"{ou_pick} de {LINEA_TOTALES} (Proy: {total_runs_int})"
                            prob_str = f"{min(99, pseudo_prob)}%"
                            score_val = pseudo_prob
                            
                        resultados_jornada.append({
                            "⏰ Hora (ET)": hora_et,
                            "✈️ Visitante": f"{e_visita} ({rec_v})",
                            "🏠 Local": f"{e_local} ({rec_l})",
                            "⚾ Abridor (V)": f"{p_visita or 'TBD'} ({whip_v:.2f})", # COLUMNA SEPARADA
                            "⚾ Abridor (L)": f"{p_local or 'TBD'} ({whip_l:.2f})", # COLUMNA SEPARADA
                            "🎯 Jugada Recomendada": jugada_str,
                            "📊 Probabilidad": prob_str,
                            "raw_time": game_dt_str or "9999-12-31T23:59:59Z",
                            "score": score_val
                        })
                        
                    resultados_jornada.sort(key=lambda x: x['raw_time'])
                    st.session_state.resultados_hoy = resultados_jornada

        if st.session_state.resultados_hoy is not None:
            if len(st.session_state.resultados_hoy) > 0:
                df_resultados = pd.DataFrame(st.session_state.resultados_hoy)
                df_display = df_resultados.drop(columns=['score', 'raw_time'], errors='ignore')
                
                # FUNCIÓN DE SEMÁFORO PARA LOS ABRIDORES
                def color_whip(row):
                    styles = [''] * len(row)
                    for j, col in enumerate(row.index):
                        if col in ['⚾ Abridor (V)', '⚾ Abridor (L)']:
                            try:
                                # Extraer numéricamente el WHIP que está entre paréntesis
                                whip_str = str(row[col]).split('(')[-1].replace(')', '')
                                whip = float(whip_str)
                                if whip < 1.00:
                                    styles[j] = 'color: #00cc66; font-weight: bold;' # Verde Élite
                                elif whip <= 1.30:
                                    styles[j] = 'color: #ff9900; font-weight: bold;' # Naranja Promedio
                                else:
                                    styles[j] = 'color: #ff4d4d; font-weight: bold;' # Rojo Deficiente
                            except:
                                pass
                    return styles
                
                df_estilizado = df_display.style\
                    .apply(color_whip, axis=1)\
                    .set_properties(**{'text-align': 'center'})\
                    .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
                
                st.dataframe(df_estilizado, use_container_width=True, hide_index=True)
                st.success("✅ Análisis completado. Cartelera lista y ordenada cronológicamente con semáforo de abridores.")
            else:
                st.info("Todos los partidos válidos de hoy ya han comenzado o finalizado.")

    with tab2:
        st.markdown("### 💣 Radar de Jonrones: Filtro de Regresión + Localía")
        st.markdown("Extrae al Top de bateadores activos HOY. Sólo muestra a los 4 mejores.")
        
        if st.button("🔍 Escanear Mercado de Jonrones (Top 4 Limpio)", type="primary", use_container_width=True):
            with st.spinner("Descargando Game Logs, aplicando exclusión por jonrones recientes y calculando métricas de poder..."):
                resultados = get_hr_hunters(anio_sel, st.session_state.fecha_hoy)
                
                if resultados:
                    st.session_state.resultados_hr = resultados
                else:
                    st.warning("No se detectaron líderes de cuadrangulares válidos para el día de hoy.")
                    
        if st.session_state.resultados_hr is not None:
            df_hr = pd.DataFrame(st.session_state.resultados_hr)
            
            df_hr_estilizado = df_hr.style\
                .set_properties(**{'text-align': 'center'})\
                .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
                
            st.dataframe(df_hr_estilizado, use_container_width=True, hide_index=True)
            st.success("✅ Análisis de Poder finalizado. Tienes frente a ti las 4 mejores opciones del día.")

    with tab3:
        st.markdown("### 🔥 Radar de Ponches: Pitcher K/9 vs Vulnerabilidad del Rival")
        st.markdown("Este motor analiza los ponches de los abridores en sus últimos 7 juegos. Te entrega al Top 4 de lanzadores con mayores probabilidades de aplastar la línea estándar de **5.5 Ponches**.")
        
        if st.button("🎯 Cazar Ponches del Día (Top 4)", type="primary", use_container_width=True):
            with st.spinner("Calculando K/9 de los abridores y procesando la vulnerabilidad ofensiva (K%) de los 30 equipos..."):
                resultados = get_strikeout_hunters(st.session_state.fecha_hoy)
                
                if resultados:
                    st.session_state.resultados_k = resultados
                else:
                    st.warning("No se detectaron abridores con suficiente información válida para proyectar hoy.")
                    
        if st.session_state.resultados_k is not None:
            df_k = pd.DataFrame(st.session_state.resultados_k)
            
            df_k_estilizado = df_k.style\
                .set_properties(**{'text-align': 'center'})\
                .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
                
            st.dataframe(df_k_estilizado, use_container_width=True, hide_index=True)
            st.success("✅ Análisis de Ponches finalizado. Estos son los 4 brazos más dominantes del día.")

    # --- VISOR DE DATOS ---
    st.markdown("---")
    with st.expander("🗃️ Explorador de Base de Datos (Auditoría Histórica)"):
        df_visor = df[['Date', 'Local', 'Visitante', 'Carreras_Local', 'Carreras_Visitante', 'Elo_L', 'Elo_V']].copy()
        df_visor['Carreras_Local'] = df_visor['Carreras_Local'].astype(int)
        df_visor['Carreras_Visitante'] = df_visor['Carreras_Visitante'].astype(int)
        df_visor['Elo_L'] = df_visor['Elo_L'].round().astype(int)
        df_visor['Elo_V'] = df_visor['Elo_V'].round().astype(int)
        
        st.dataframe(df_visor, use_container_width=True, hide_index=True)
else:
    st.info("👈 Presiona 'Descargar Historial Base' en la barra lateral para encender el motor predictivo")
