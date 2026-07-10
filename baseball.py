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
st.markdown("Proyección Sabermétrica: Elo, Racha, H2H, Pitagórico, Rendimiento Dividido, Abridores, Totales y Jonrones")
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

def get_pitcher_era(pitcher_name):
    if not pitcher_name or pitcher_name == 'TBD': 
        return 4.50
    try:
        players = statsapi.lookup_player(pitcher_name)
        if not players: return 4.50
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
                                
                                total_er = 0
                                total_outs = 0
                                
                                for game in last_7:
                                    g_stats = game.get('stat', {})
                                    total_er += int(g_stats.get('earnedRuns', 0))
                                    ip_str = str(g_stats.get('inningsPitched', '0.0'))
                                    if '.' in ip_str:
                                        full, frac = ip_str.split('.')
                                        total_outs += (int(full) * 3) + int(frac)
                                    else:
                                        total_outs += int(ip_str) * 3
                                        
                                if total_outs > 0:
                                    era = (total_er / (total_outs / 3.0)) * 9.0
                                    return round(era, 2)
                                else:
                                    return 4.50
        except Exception:
            pass
            
        try:
            stats_season = statsapi.player_stat_data(player_id, group="pitching", type="season")
            if stats_season and 'stats' in stats_season and len(stats_season['stats']) > 0:
                era_str = stats_season['stats'][0]['stats'].get('era', '-.--')
                if era_str != '-.--':
                    return float(era_str)
        except Exception:
            pass
                
        return 4.50
    except:
        return 4.50

# NUEVO MOTOR OPTIMIZADO: Cazador de Jonrones (Top 4 Estricto y limpio)
def get_hr_hunters(anio, fecha_hoy):
    try:
        # 1. Mapear qué equipos juegan hoy y si son Locales o Visitantes
        juegos_hoy = statsapi.schedule(date=fecha_hoy, sportId=1)
        equipos_hoy = {}
        for juego in juegos_hoy:
            if juego.get('status', '') not in ['Postponed', 'Cancelled']:
                equipos_hoy[juego.get('home_id')] = 'Local'
                equipos_hoy[juego.get('away_id')] = 'Visitante'
        
        # 2. Descargar Top 40 para asegurar que al menos queden suficientes activos hoy
        data = statsapi.get('stats_leaders', {'leaderCategories': 'homeRuns', 'season': anio, 'limit': 40, 'statGroup': 'hitting'})
        if not data or 'leagueLeaders' not in data or len(data['leagueLeaders']) == 0:
            return []
            
        leaders = data['leagueLeaders'][0].get('leaders', [])
        
        # 3. Filtrar estrictamente a los jugadores que tienen juego HOY y extraer su equipo
        jugadores_activos = []
        for p in leaders:
            team_id = p.get('team', {}).get('id')
            team_name = p.get('team', {}).get('name', 'Unknown')
            if team_id in equipos_hoy:
                p['condicion_hoy'] = equipos_hoy[team_id]
                p['team_name'] = team_name
                jugadores_activos.append(p)
                
        resultados = []
        
        # 4. Procesar estadísticas de los jugadores activos
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
            
            for block in stats_blocks:
                if block.get('type', {}).get('displayName') == 'season':
                    season_ab = max(1, int(block.get('splits', [{}])[0].get('stat', {}).get('atBats', 1)))
                elif block.get('type', {}).get('displayName') == 'gameLog':
                    splits = block.get('splits', [])
                    splits.sort(key=lambda x: x.get('date', ''), reverse=True)
                    last_10 = splits[:10]
                    
                    for game in last_10:
                        g_stats = game.get('stat', {})
                        l10_hr += int(g_stats.get('homeRuns', 0))
                        l10_ab += int(g_stats.get('atBats', 0))
            
            l10_ab = max(1, l10_ab) # Evitar división por cero
            
            # FÓRMULA SABERMÉTRICA (Índice de Temperatura + Bono de Localía)
            freq_season = season_hr / season_ab
            freq_recent = l10_hr / l10_ab
            
            bono_localia = 1.10 if condicion == 'Local' else 1.0 # +10% de boost si batea en casa
            
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
            
        # 5. ORDENAR Y RECORTAR AL TOP 4 EXACTO
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

# --- ESTADO DE SESIÓN ---
if 'df_mlb' not in st.session_state: st.session_state.df_mlb = None
if 'fecha_hoy' not in st.session_state: st.session_state.fecha_hoy = datetime.date.today().strftime('%Y-%m-%d')
if 'resultados_hoy' not in st.session_state: st.session_state.resultados_hoy = None
if 'resultados_hr' not in st.session_state: st.session_state.resultados_hr = None

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
    
    tab1, tab2 = st.tabs(["📅 Cartelera Automática (Juegos de Hoy)", "💣 Caza-Jonrones (Prop Bets)"])
    
    with tab1:
        st.markdown(f"### 🎯 Partidos programados para hoy: **{st.session_state.fecha_hoy}**")
        
        if st.button("⚡ Analizar Jornada Completa", type="primary", use_container_width=True):
            with st.spinner("Escaneando el calendario, calculando ERA real (L7) de abridores y evaluando Best Bets..."):
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
                            try: hora_et = pd.to_datetime(game_dt_str).tz_convert('US/Eastern').strftime('%I:%M %p')
                            except: hora_et = 'TBD'
                        else: hora_et = 'TBD'
                            
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
                        
                        era_l = get_pitcher_era(p_local)
                        era_v = get_pitcher_era(p_visita)
                        
                        prob = clf.predict_proba(np.array([[elo_l, elo_v]]))[0][1]
                        
                        pitcher_adj = ((era_v - era_l) / 9.0) * 5.0 * 0.10
                        
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
                        
                        adj_runs_l = (era_v - 4.50) * (5.0/9.0)
                        adj_runs_v = (era_l - 4.50) * (5.0/9.0)
                        
                        c_l = max(0.5, c_l + adj_runs_l)
                        c_v = max(0.5, c_v + adj_runs_v)
                        
                        c_l_int = int(round(c_l))
                        c_v_int = int(round(c_v))
                        
                        if prob_final_local > 0.5 and c_l_int <= c_v_int:
                            c_l_int = c_v_int + 1
                        elif prob_final_local <= 0.5 and c_v_int <= c_l_int:
                            c_v_int = c_l_int + 1
                            
                        total_runs = round(c_v + c_l, 2)
                        
                        if total_runs > LINEA_TOTALES: ou_pick = "🔥 ALTA"
                        else: ou_pick = "🧊 BAJA"
                        
                        resultados_jornada.append({
                            "⏰ Hora (ET)": hora_et,
                            "✈️ Visitante": f"{e_visita} ({rec_v})",
                            "🏠 Local": f"{e_local} ({rec_l})",
                            "⚾ Abridores (L7 ERA)": f"{p_visita or 'TBD'} ({era_v:.2f}) vs {p_local or 'TBD'} ({era_l:.2f})",
                            "🏆 Proyección": ganador,
                            "📊 Prob.": f"{pct_final}%",
                            "🎯 Marcador Proy.": f"{c_v_int} - {c_l_int}",
                            f"⚖️ Total ({LINEA_TOTALES})": f"{ou_pick} ({total_runs:.2f})"
                        })
                    st.session_state.resultados_hoy = resultados_jornada

        if st.session_state.resultados_hoy is not None:
            if len(st.session_state.resultados_hoy) > 0:
                df_resultados = pd.DataFrame(st.session_state.resultados_hoy)
                col_total_name = f"⚖️ Total ({LINEA_TOTALES})"
                
                todas_las_jugadas = []
                for i, row in df_resultados.iterrows():
                    ganador = row['🏆 Proyección']
                    prob = int(str(row['📊 Prob.']).replace('%', ''))
                    is_visita = ganador in row['✈️ Visitante']
                    
                    try: total_val = float(str(row[col_total_name]).split('(')[1].replace(')', ''))
                    except: total_val = LINEA_TOTALES
                    
                    if (is_visita and prob >= 55) or (not is_visita and prob >= 65):
                        todas_las_jugadas.append({'row': i, 'col': '🏆 Proyección', 'score': prob})
                        
                    diff_total = abs(total_val - LINEA_TOTALES)
                    if diff_total >= 1.0: 
                        pseudo_prob = 50 + (diff_total * 10)
                        todas_las_jugadas.append({'row': i, 'col': col_total_name, 'score': pseudo_prob})
                        
                top_3_jugadas = sorted(todas_las_jugadas, key=lambda x: x['score'], reverse=True)[:3]
                top_3_coordenadas = [(p['row'], p['col']) for p in top_3_jugadas]
                
                def aplicar_semaforo_top3(row):
                    styles = [''] * len(row)
                    idx = row.name
                    ganador = row['🏆 Proyección']
                    prob = int(str(row['📊 Prob.']).replace('%', ''))
                    is_visita = ganador in row['✈️ Visitante']
                    
                    try: total_val = float(str(row[col_total_name]).split('(')[1].replace(')', ''))
                    except: total_val = LINEA_TOTALES
                    diff_total = abs(total_val - LINEA_TOTALES)
                    
                    for j, col in enumerate(row.index):
                        if col == '🏆 Proyección':
                            if (idx, col) in top_3_coordenadas:
                                styles[j] = 'background-color: #198754; color: white; font-weight: bold;'
                            elif (is_visita and prob >= 55) or (not is_visita and prob >= 65):
                                styles[j] = 'background-color: #ffc107; color: black; font-weight: bold;'
                                
                        elif col == col_total_name:
                            if (idx, col) in top_3_coordenadas:
                                styles[j] = 'background-color: #198754; color: white; font-weight: bold;'
                            elif diff_total >= 1.0:
                                styles[j] = 'background-color: #ffc107; color: black; font-weight: bold;'
                                
                    return styles

                df_estilizado = df_resultados.style.apply(aplicar_semaforo_top3, axis=1)
                st.dataframe(df_estilizado, use_container_width=True, hide_index=True)
                st.success("✅ Análisis completado | 🟡 Alta Confianza | 🟢 Top 3 Mejores Jugadas (Best Bets)")
            else:
                st.info("Todos los partidos válidos de hoy ya han comenzado o finalizado.")

    with tab2:
        st.markdown("### 💣 Radar de Jonrones: Localía vs. Temperatura Actual")
        st.markdown("Filtra automáticamente a los líderes jonroneros que **sí juegan el día de hoy** y evalúa sus últimos 10 juegos. El motor inyecta un **+10% de bono a los bateadores locales** y te entrega únicamente el **Top 4 definitivo** listo para apostar.")
        
        if st.button("🔍 Escanear Mercado de Jonrones (Jugadores Activos Hoy)", type="primary", use_container_width=True):
            with st.spinner("Conectando con el calendario MLB de hoy... cruzando datos y aislando al Top 4 de élite..."):
                resultados = get_hr_hunters(anio_sel, st.session_state.fecha_hoy)
                
                if resultados:
                    st.session_state.resultados_hr = resultados
                else:
                    st.warning("No se detectaron líderes de cuadrangulares con juegos programados para el día de hoy o hubo un error de conexión.")
                    
        if st.session_state.resultados_hr is not None:
            df_hr = pd.DataFrame(st.session_state.resultados_hr)
            st.dataframe(df_hr, use_container_width=True, hide_index=True)
            st.success("✅ Calendario sincronizado y análisis completado. Top 4 mejores opciones de cuadrangular listas.")

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