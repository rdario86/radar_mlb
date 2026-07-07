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
st.markdown("Proyección Sabermétrica: Elo, Racha, H2H, Pitagórico, Rendimiento Dividido y Totales Híbridos")
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

# Teorema Pitagórico
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

# Rendimiento Dividido
def get_splits_win_pct(home_team, away_team, df):
    home_games = df[df['Local'] == home_team]
    home_win_pct = sum(1 for _, row in home_games.iterrows() if row['Carreras_Local'] > row['Carreras_Visitante']) / len(home_games) if len(home_games) > 0 else 0.5
    
    away_games = df[df['Visitante'] == away_team]
    away_win_pct = sum(1 for _, row in away_games.iterrows() if row['Carreras_Visitante'] > row['Carreras_Local']) / len(away_games) if len(away_games) > 0 else 0.5
    
    return home_win_pct, away_win_pct

# Proyección Híbrida
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

def aplicar_semaforo_confianza(row):
    styles = [''] * len(row)
    ganador = row['🏆 Proyección']
    prob = int(str(row['📊 Prob.']).replace('%', ''))
    
    # Extraer el valor total proyectado de la celda
    col_total = f"⚖️ Total ({LINEA_TOTALES})"
    try:
        total_val = float(str(row[col_total]).split('(')[1].replace(')', ''))
    except:
        total_val = LINEA_TOTALES
    
    for i, col in enumerate(row.index):
        # Semáforo para Ganador
        if col == '🏆 Proyección':
            if (ganador in row['✈️ Visitante'] and prob >= 55) or (ganador in row['🏠 Local'] and prob >= 65):
                styles[i] = 'background-color: #198754; color: white; font-weight: bold;'
        
        # Semáforo para Totales (Alta/Baja)
        elif col == col_total:
            if total_val <= 7.5 or total_val >= 9.5:
                styles[i] = 'background-color: #198754; color: white; font-weight: bold;'
                
    return styles

# --- ESTADO DE SESIÓN ---
if 'df_mlb' not in st.session_state: st.session_state.df_mlb = None
if 'fecha_hoy' not in st.session_state: st.session_state.fecha_hoy = datetime.date.today().strftime('%Y-%m-%d')

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
                batch = statsapi.schedule(start_date=start, end_date=end, sportId=1)
                if batch: data_total.extend(batch)
                time.sleep(0.5)
            
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
        except Exception as e: st.sidebar.error(f"Error: {e}")

# --- ÁREA PRINCIPAL ---
if st.session_state.df_mlb is not None:
    df = st.session_state.df_mlb.copy()
    
    df['Win'] = (df['Carreras_Local'] > df['Carreras_Visitante']).astype(int)
    clf = RandomForestClassifier(max_depth=MAX_DEPTH_ELO, random_state=42).fit(df[['Elo_L', 'Elo_V']], df['Win'])
    
    tab1, tab2 = st.tabs(["📅 Cartelera Automática (Hoy)", "🔍 Análisis Manual"])
    
    with tab1:
        st.markdown(f"### 🎯 Partidos programados para hoy: **{st.session_state.fecha_hoy}**")
        
        if st.button("⚡ Analizar Jornada Completa", type="primary", use_container_width=True):
            with st.spinner("Escaneando el calendario y calculando proyecciones con matriz pitagórica y totales..."):
                juegos_hoy = statsapi.schedule(date=st.session_state.fecha_hoy, sportId=1)
                
                if not juegos_hoy:
                    st.warning("No hay juegos oficiales programados para la fecha de hoy.")
                else:
                    resultados_jornada = []
                    equipos_procesados = set() 
                    
                    for juego in juegos_hoy:
                        e_local = juego['home_name']
                        e_visita = juego['away_name']
                        
                        if e_local not in MLB_TEAM_WHITELIST or e_visita not in MLB_TEAM_WHITELIST:
                            continue
                            
                        if e_local in equipos_procesados or e_visita in equipos_procesados:
                            continue
                        
                        equipos_procesados.add(e_local)
                        equipos_procesados.add(e_visita)
                            
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
                        
                        prob = clf.predict_proba(np.array([[elo_l, elo_v]]))[0][1]
                        prob_final_local = (prob + 
                                            (racha_l - racha_v) * PESO_RACHA + 
                                            (h2h - 0.5) * PESO_H2H + 
                                            (luck_l - luck_v) * PESO_PITAGORICO + 
                                            (split_l - split_v) * PESO_SPLITS)
                        
                        if prob_final_local > 0.5:
                            ganador = e_local
                            pct_bruto = prob_final_local
                        else:
                            ganador = e_visita
                            pct_bruto = 1.0 - prob_final_local
                            
                        pct_final = int(round(max(min(pct_bruto, 0.99), 0.01) * 100))
                        
                        c_v, c_l = get_hybrid_run_projection(e_visita, e_local, df)
                        total_runs = c_v + c_l
                        
                        if prob_final_local > 0.5 and c_l <= c_v:
                            c_l = round((total_runs / 2) + 0.25, 2)
                            c_v = round((total_runs / 2) - 0.25, 2)
                        elif prob_final_local <= 0.5 and c_v <= c_l:
                            c_v = round((total_runs / 2) + 0.25, 2)
                            c_l = round((total_runs / 2) - 0.25, 2)
                            
                        total_runs = round(c_v + c_l, 2)
                        
                        if total_runs > LINEA_TOTALES: ou_pick = "🔥 ALTA"
                        else: ou_pick = "🧊 BAJA"
                        
                        resultados_jornada.append({
                            "✈️ Visitante": f"{e_visita} ({rec_v})",
                            "🏠 Local": f"{e_local} ({rec_l})",
                            "🏆 Proyección": ganador,
                            "📊 Prob.": f"{pct_final}%",
                            "🎯 Marcador Proy.": f"{c_v:.2f} - {c_l:.2f}",
                            f"⚖️ Total ({LINEA_TOTALES})": f"{ou_pick} ({total_runs:.2f})"
                        })
                    
                    if resultados_jornada:
                        df_resultados = pd.DataFrame(resultados_jornada)
                        df_estilizado = df_resultados.style.apply(aplicar_semaforo_confianza, axis=1)
                        st.dataframe(df_estilizado, use_container_width=True, hide_index=True)
                        st.success("✅ Análisis completado. Jugadas de alta confianza resaltadas en verde")
                    else:
                        st.info("No se encontraron partidos válidos en la lista oficial para hoy.")

    with tab2:
        st.markdown("### 🔍 Análisis Detallado (Partido Único)")
        equipos = sorted(list(set(df['Local'].unique()) | set(df['Visitante'].unique())))
        
        c1, c2 = st.columns(2)
        with c1: e_local_manual = st.selectbox("🏠 Equipo Local:", equipos, key="loc_man")
        with c2: e_visita_manual = st.selectbox("✈️ Equipo Visitante:", [e for e in equipos if e != e_local_manual], key="vis_man")

        if st.button("🚀 Lanzar Proyección Individual"):
            rec_l = get_team_record(e_local_manual, df)
            rec_v = get_team_record(e_visita_manual, df)
            
            elo_l = df[df['Local'] == e_local_manual].tail(1)['Elo_L'].values[0] if len(df[df['Local'] == e_local_manual]) > 0 else 1500
            elo_v = df[df['Visitante'] == e_visita_manual].tail(1)['Elo_V'].values[0] if len(df[df['Visitante'] == e_visita_manual]) > 0 else 1500
            
            elo_l += 35 
            
            racha_l = get_recent_form(e_local_manual, df)
            racha_v = get_recent_form(e_visita_manual, df)
            h2h = get_h2h_wins(e_local_manual, e_visita_manual, df)
            luck_l = get_pythagorean_luck(e_local_manual, df)
            luck_v = get_pythagorean_luck(e_visita_manual, df)
            split_l, split_v = get_splits_win_pct(e_local_manual, e_visita_manual, df)
            
            prob = clf.predict_proba(np.array([[elo_l, elo_v]]))[0][1]
            prob_final_local = (prob + 
                                (racha_l - racha_v) * PESO_RACHA + 
                                (h2h - 0.5) * PESO_H2H + 
                                (luck_l - luck_v) * PESO_PITAGORICO + 
                                (split_l - split_v) * PESO_SPLITS)
            
            if prob_final_local > 0.5:
                ganador = e_local_manual
                porcentaje_bruto = prob_final_local
            else:
                ganador = e_visita_manual
                porcentaje_bruto = 1.0 - prob_final_local
                
            porcentaje_entero = int(round(max(min(porcentaje_bruto, 0.99), 0.01) * 100))
                      
            c_v, c_l = get_hybrid_run_projection(e_visita_manual, e_local_manual, df)
            total_runs = c_v + c_l
            
            if prob_final_local > 0.5 and c_l <= c_v:
                c_l = round((total_runs / 2) + 0.25, 2)
                c_v = round((total_runs / 2) - 0.25, 2)
            elif prob_final_local <= 0.5 and c_v <= c_l:
                c_v = round((total_runs / 2) + 0.25, 2)
                c_l = round((total_runs / 2) - 0.25, 2)
                
            total_runs = round(c_v + c_l, 2)
            
            if total_runs > LINEA_TOTALES: ou_pick = "🔥 ALTA"
            else: ou_pick = "🧊 BAJA"
            
            st.markdown("### 📊 Pizarra de Proyección")
            
            # Alerta de confianza para el ganador
            if (ganador == e_visita_manual and porcentaje_entero >= 55) or (ganador == e_local_manual and porcentaje_entero >= 65):
                st.success("🟢 ESTA PROYECCIÓN DE GANADOR CUMPLE CON EL UMBRAL DE ALTA CONFIANZA")
                
            st.metric("🏆 Ganador Proyectado", ganador, f"{porcentaje_entero}%")
            
            st.markdown("#### ⚖️ Mercado de Totales (Over/Under)")
            
            # Alerta de confianza para Totales (Alta/Baja)
            if total_runs <= 7.5 or total_runs >= 9.5:
                st.success("🟢 ESTA PROYECCIÓN DE TOTALES CUMPLE CON EL UMBRAL DE ALTA CONFIANZA")
                
            k1, k2, k3 = st.columns(3)
            k1.metric(f"✈️ {e_visita_manual} (Visita)", f"{c_v:.2f} Carreras")
            k2.metric(f"🏠 {e_local_manual} (Local)", f"{c_l:.2f} Carreras")
            k3.metric(f"Línea de Las Vegas: {LINEA_TOTALES}", f"Carreras Proyectadas: {total_runs:.2f}")

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