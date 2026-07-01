import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from procesador import (
    procesar_cromatograma, 
    analizar_color_hsv_zonas, 
    analizar_orillas_y_enzimas, 
    analizar_plumas_cresta
)

st.set_page_config(layout="wide", page_title="SoilAnalytica Pro")

if 'historial' not in st.session_state:
    st.session_state.historial = {}
if 'comparador' not in st.session_state:
    st.session_state.comparador = []

# Estilo profesional
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    footer {visibility: hidden;}
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 4px; padding: 4px 16px; }
    .badge-optimo {
        background-color: #1e6b3b;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        margin-left: 8px;
    }
    .badge-medio {
        background-color: #d4a017;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        margin-left: 8px;
    }
    .badge-limitante {
        background-color: #a83232;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        margin-left: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

def local_binary_pattern_manual(img, P=8, R=1):
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    angles = 2 * np.pi * np.arange(P) / P
    offsets_x = np.round(R * np.cos(angles)).astype(int)
    offsets_y = np.round(R * np.sin(angles)).astype(int)
    
    for i in range(R, h - R):
        for j in range(R, w - R):
            center = img[i, j]
            code = 0
            for k in range(P):
                x = j + offsets_x[k]
                y = i + offsets_y[k]
                if img[y, x] >= center:
                    code |= (1 << k)
            lbp[i, j] = code
    return lbp

def obtener_estado(valor, umbral_optimo, umbral_medio):
    if valor >= umbral_optimo:
        return "Óptimo", "badge-optimo"
    elif valor >= umbral_medio:
        return "Medio", "badge-medio"
    else:
        return "Limitante", "badge-limitante"

def analizar_radial_preciso(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    gray_smooth = cv2.bilateralFilter(gray_eq, 9, 75, 75)
    
    h, w = gray_smooth.shape
    cy, cx = h // 2, w // 2
    max_r = int(min(cy, cx) * 0.82)
    
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    mask_oxi = dist < (max_r * 0.12)
    mask_min = (dist >= max_r * 0.35) & (dist <= max_r * 0.68)
    mask_bio = (dist >= max_r * 0.76) & (dist <= max_r * 0.94)
    
    intensidad_oxi = np.mean(gray_smooth[mask_oxi]) if np.any(mask_oxi) else 128
    intensidad_min = np.mean(gray_smooth[mask_min]) if np.any(mask_min) else 128
    textura_bio = np.std(gray_smooth[mask_bio]) if np.any(mask_bio) else 30
    
    max_intensity = np.max(gray_smooth)
    min_intensity = np.min(gray_smooth)
    rango = max_intensity - min_intensity if max_intensity > min_intensity else 1
    
    oxi_raw = 40 + ((intensidad_oxi - min_intensity) / rango) * 55
    min_raw = 42 + ((intensidad_min - min_intensity) / rango) * 52
    bio_raw = 48 + (min(1.0, textura_bio / 58.0)) * 42
    
    oxigenacion = np.clip(oxi_raw * 0.88 - 2.5, 45, 85)
    mineralizacion = np.clip(min_raw * 1.05 + 2.8, 50, 88)
    biologia = np.clip(bio_raw * 0.89 - 2.0, 48, 84)
    
    rad_vals = []
    paso = max_r / 150
    for r in np.arange(0, max_r, paso):
        mask_ring = (dist >= r) & (dist < r + paso)
        if np.any(mask_ring):
            rad_vals.append(float(np.mean(gray_smooth[mask_ring])))
        else:
            rad_vals.append(0.0)
    
    tiene_anillos = False
    if len(rad_vals) > 60:
        autcorr = np.correlate(rad_vals, rad_vals, mode='full')
        autcorr = autcorr[len(autcorr)//2:len(autcorr)//2 + len(rad_vals)//2]
        if len(autcorr) > 30:
            max_autcorr = np.max(autcorr[15:min(60, len(autcorr))])
            tiene_anillos = max_autcorr > np.mean(autcorr[20:]) * 1.25
            
    # LO QUE SE AGREGÓ: Procesamiento real de nuevos módulos
    datos_color = analizar_color_hsv_zonas(imagen, cy, cx, max_r)
    datos_orilla = analizar_orillas_y_enzimas(imagen, cy, cx, max_r)
    datos_plumas = analizar_plumas_cresta(imagen, cy, cx, max_r)
    
    return {
        'oxigenacion': round(oxigenacion, 1),
        'mineralizacion': round(mineralizacion, 1),
        'biologia': round(biologia, 1),
        'perfil_radial': rad_vals,
        'tiene_anillos': tiene_anillos,
        'max_r': max_r,
        'paso': paso,
        # LO QUE SE AGREGÓ: Campos nuevos mapeados al retorno
        'mineral_predominante': datos_color['mineral_predominante'],
        'actividad_enzimatica': datos_orilla['actividad_enzimatica'],
        'patron_plumas': datos_plumas['patron_plumas']
    }

def detectar_caracteristicas(imagen, analisis):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cy, cx = h//2, w//2
    
    caracteristicas = []
    
    try:
        angulos = np.linspace(0, 2*np.pi, 24, endpoint=False)
        intensidades = []
        for ang in angulos:
            x = int(cx + analisis['max_r'] * 0.5 * np.cos(ang))
            y = int(cy + analisis['max_r'] * 0.5 * np.sin(ang))
            if 0 <= x < w and 0 <= y < h:
                intensidades.append(gray[y, x])
        if intensidades:
            simetria = 1 - (np.std(intensidades) / (np.mean(intensidades) + 1e-6))
            if simetria > 0.82:
                caracteristicas.append("Alta simetría radial - desarrollo equilibrado")
            elif simetria < 0.58:
                caracteristicas.append("Asimetría significativa - posible estrés direccional")
    except:
        pass
    
    if analisis['tiene_anillos']:
        caracteristicas.append("Anillos concéntricos visibles - procesos sucesionales definidos")
    
    return caracteristicas

def recomendar_organismos(oxi, minr, bio):
    if oxi >= 68 and minr >= 70 and bio >= 72:
        return """
        **Microorganismos benéficos sugeridos:**
        • *Trichoderma spp.* - Control biológico y BAM de materia orgánica
        • *Pseudomonas fluorescens* - Promotor de crecimiento y solubilización de fósforo
        • *Micorrizas arbusculares (Glomus spp.)* - Aumentan absorción de nutrientes
        • *Rhizobium spp.* - Fijación de nitrógeno (en leguminosas)
        • *Bacillus subtilis* - Producción de fitohormonas y antibióticos naturales
        """
    elif oxi < 58:
        return """
        **Microorganismos para suelos con compactación:**
        • *Bacillus megaterium* - Tolera baja oxigenación, solubiliza fósforo
        • *Lactobacillus spp.* - Previene patógenos en condiciones de saturación
        • *Saccharomyces cerevisiae* - Mejora estructura en condiciones difíciles
        • *Azospirillum brasilense* - Fija nitrógeno en condiciones subóptimas
        """
    elif minr < 60:
        return """
        **Microorganismos para activar mineralización:**
        • *Trichoderma harzianum* - Acelera descomposición de residuos
        • *Cellulomonas spp.* - Degrada celulosa y libera carbono lábil
        • *Streptomyces spp.* - Descompone materiales recalcitrantes
        • *Aspergillus niger* - Solubiliza fósforo y potasio de minerales
        """
    elif bio < 62:
        return """
        **Microorganismos para reactivar biología:**
        • *Micorrizas arbusculares (Rhizophagus irregularis)* - Colonización radical prioritaria
        • *Pseudomonas putida* - Producción de sideróforos y fitohormonas
        • *Trichoderma koningii* - Estimula diversidad fúngica nativa
        • *Bacillus amyloliquefaciens* - Alto poder de colonización y antibiosis
        """
    else:
        return """
        **Microorganismos para mantener y potenciar:**
        • *Micorrizas nativas* - Preservar mediante rotación de cultivos
        • *Rhizobium* específicos para leguminosas del sistema
        • *Azotobacter chroococcum* - Fijación de nitrógeno de vida libre
        • *Bacillus pumilus* - Producción de enzimas extracelulares
        """

def generar_informe(id_m, analisis, caracteristicas):
    oxi = analisis['oxigenacion']
    bio = analisis['biologia']
    minr = analisis['mineralizacion']
    
    if oxi >= 72:
        txt_oxi = f"Oxigenación: {oxi}% | Núcleo claro y bien aireado. La estructura porosa permite un intercambio gaseoso óptimo, facilitando la respiración radicular y la actividad aeróbica."
    elif oxi >= 62:
        txt_oxi = f"Oxigenación: {oxi}% | Núcleo parcialmente funcional. Se observa una densificación moderada que podría limitar la difusión de oxígeno en condiciones de humedad elevada."
    elif oxi >= 52:
        txt_oxi = f"Oxigenación: {oxi}% | Núcleo reducido o compactado. La estructura presenta limitaciones significativas para el intercambio gaseoso, favoreciendo condiciones anaeróbicas."
    else:
        txt_oxi = f"Oxigenación: {oxi}% | Núcleo denso u oscuro. Existe un déficit crítico de aireación que afecta negativamente los procesos biológicos y químicos del suelo."
    
    if minr >= 74:
        txt_min = f"Mineralización: {minr}% | Fase mineral bien integrada. Los nutrientes se encuentran disponibles en la matriz orgánico-mineral, con una cinética de liberación equilibrada."
    elif minr >= 64:
        txt_min = f"Mineralización: {minr}% | Proceso activo de transformación. Se detectan compuestos orgánicos en transición hacia formas asimilables, con una integración mineral en desarrollo."
    elif minr >= 54:
        txt_min = f"Mineralización: {minr}% | Mineralización incipiente. Predominan formas orgánicas no humificadas que requieren mayor tiempo de descomposición biológica."
    else:
        txt_min = f"Mineralización: {minr}% | Baja evolución mineral. El material orgánico presenta escasa integración, posiblemente por déficit de actividad microbiana o relación C/N inadecuada."
    
    if bio >= 76:
        txt_bio = f"Biología: {bio}% | Actividad biológica sobresaliente. Se observan redes de hifas fúngicas, microagregados bacterianos y una estructura externa rugosa indicativa de alta colonización microbiana."
    elif bio >= 64:
        txt_bio = f"Biología: {bio}% | Comunidad microbiana funcional. La zona externa muestra estructuras filamentosas visibles aunque con cierta fragmentación, indicando actividad estable pero mejorable."
    elif bio >= 52:
        txt_bio = f"Biología: {bio}% | Población microbiana moderada. Se detectan colonias puntuales pero con baja densidad de red, sugiriendo estrés ambiental o limitación nutricional."
    else:
        txt_bio = f"Biología: {bio}% | Baja actividad biológica. La zona externa se presenta lisa o con escasa rugosidad, indicando déficit de colonización fúngica y bacteriana."
    
    if oxi >= 68 and minr >= 70 and bio >= 72:
        diagnosis = "Suelo equilibrado - Las tres fases se encuentran sincronizadas, indicando condiciones óptimas."
        manejo = "Mantener prácticas actuales. Monitoreo semestral."
    elif oxi >= 60 and minr >= 65 and bio >= 65:
        diagnosis = "Perfil en desarrollo - Evolución positiva con margen de mejora."
        manejo = "Continuar con manejo orgánico and monitorear evolución."
    elif oxi < 58:
        diagnosis = "Limitante primario: Oxigenación - Compactación o saturación del núcleo."
        manejo = "Mejorar estructura con materia orgánica gruesa y evitar compactación."
    elif minr < 60:
        diagnosis = "Limitante primario: Mineralización - Procesos de transformación lentos."
        manejo = "Ajustar relación C/N con compost maduro."
    elif bio < 62:
        diagnosis = "Limitante primario: Biología - Comunidad microbiana poco activa."
        manejo = "Inocular con microorganismos benéficos y mantener cobertura vegetal."
    else:
        diagnosis = "Perfil asimétrico - Desbalance entre componentes evaluados."
        manejo = "Fortalecer el indicador de menor valor."
        
    microorganismos = recomendar_organismos(oxi, minr, bio)
    
    if caracteristicas:
        extra = f"\n\n**Hallazgos adicionales:** {', '.join(caracteristicas[:2])}"
    else:
        extra = ""
        
    conclusion = f"""
    **Diagnóstico:** {diagnosis}
    
    **Recomendación:** {manejo}
    
    {microorganismos}
    {extra}
    """
    
    return txt_oxi, txt_min, txt_bio, conclusion

# Sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("🔵 **Developed by ArmasG**")
    st.markdown("---")
    
    st.header("Muestras")
    
    for id_h in list(st.session_state.historial.keys()):
        col1, col2 = st.columns([4,1])
        with col1:
            if st.button(f"{id_h}", key=f"btn_{id_h}", use_container_width=True):
                st.session_state.id_actual = id_h
        with col2:
            if st.button("+", key=f"comp_{id_h}"):
                if id_h not in st.session_state.comparador:
                    st.session_state.comparador.append(id_h)
                else:
                    st.session_state.comparador.remove(id_h)
    
    if st.session_state.comparador:
        st.divider()
        st.caption("Comparando: " + ", ".join(st.session_state.comparador))
        
    st.divider()
    st.caption("Laboratorio de Suelos")
    st.caption("v1.0 | Aprendizaje con muestras reales")

# Main app
st.title("SoilAnalytica Pro")
st.caption("ANÁLISIS CUANTITATIVO DE CROMATOGRAFÍAS DE SUELO")

col_id, col_file = st.columns([1, 2])
with col_id:
    id_input = st.text_input("ID de la muestra", placeholder="Ej: M1")
with col_file:
    archivo = st.file_uploader("Cargar imagen del cromatograma", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if archivo and id_input:
    if id_input not in st.session_state.historial:
        with open("temp.jpg", "wb") as f:
            f.write(archivo.getbuffer())
        
        orig, cromo = procesar_cromatograma("temp.jpg")
        
        if cromo is not None:
            with st.spinner("Procesando cromatograma..."):
                analisis = analizar_radial_preciso(cromo)
                caract = detectar_caracteristicas(cromo, analisis)
                
                st.session_state.historial[id_input] = {
                    "img": cv2.cvtColor(cromo, cv2.COLOR_BGR2RGB),
                    "analisis": analisis,
                    "caract": caract
                }
                st.session_state.id_actual = id_input
                st.rerun()

# Resultados
if 'id_actual' in st.session_state:
    datos = st.session_state.historial.get(st.session_state.id_actual)
    if datos:
        analisis = datos["analisis"]
        
        UMBRAL_OXI_OPTIMO = 68
        UMBRAL_OXI_MEDIO = 58
        UMBRAL_MIN_OPTIMO = 70
        UMBRAL_MIN_MEDIO = 60
        UMBRAL_BIO_OPTIMO = 72
        UMBRAL_BIO_MEDIO = 60
        
        estado_oxi, class_oxi = obtener_estado(analisis['oxigenacion'], UMBRAL_OXI_OPTIMO, UMBRAL_OXI_MEDIO)
        estado_min, class_min = obtener_estado(analisis['mineralizacion'], UMBRAL_MIN_OPTIMO, UMBRAL_MIN_MEDIO)
        estado_bio, class_bio = obtener_estado(analisis['biologia'], UMBRAL_BIO_OPTIMO, UMBRAL_BIO_MEDIO)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(datos["img"], use_container_width=True)
        
        with col2:
            st.subheader("Métricas cuantitativas")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 14px; color: #aaa;">Oxigenación</div>
                    <div style="font-size: 32px; font-weight: bold;">{analisis['oxigenacion']}%</div>
                    <div class="{class_oxi}" style="margin-top: 5px;">{estado_oxi}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 14px; color: #aaa;">Mineralización</div>
                    <div style="font-size: 32px; font-weight: bold;">{analisis['mineralizacion']}%</div>
                    <div class="{class_min}" style="margin-top: 5px;">{estado_min}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 14px; color: #aaa;">Biología</div>
                    <div style="font-size: 32px; font-weight: bold;">{analisis['biologia']}%</div>
                    <div class="{class_bio}" style="margin-top: 5px;">{estado_bio}</div>
                </div>
                """, unsafe_allow_html=True)
            
            fig = go.Figure(go.Scatterpolar(
                r=[analisis['oxigenacion'], analisis['biologia'], analisis['mineralizacion'], analisis['oxigenacion']],
                theta=['Oxigenación', 'Biología', 'Mineralización', 'Oxigenación'],
                fill='toself', fillcolor='rgba(46, 204, 113, 0.2)',
                line=dict(color='#2ecc71', width=2)
            ))
            fig.update_layout(
                height=350,
                margin=dict(l=40, r=40, t=25, b=25),
                polar=dict(radialaxis=dict(range=[0, 100], tickfont=dict(size=9)))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
       # =========================================================================
        # SECCIÓN REDISEÑADA: ANÁLISIS AVANZADO Y GEOLOCALIZACIÓN SATELITAL
        # =========================================================================
        st.markdown("### 🔍 Análisis Morfológico, Cromático y Georreferenciación")
        
        col_bloque_izq, col_bloque_der = st.columns([1, 1])
        
        with col_bloque_izq:
            st.markdown("<p style='color: #aaa; font-size: 13px; font-weight: bold; margin-bottom: 8px;'>DIAGNÓSTICO FORMAL DE MATRIZ</p>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 12px;">
                <span style="color: #58a6ff; font-weight: bold;">🪶 Análisis de Plumas:</span> 
                <div style="color: #e6edf3; font-size: 14px; margin-top: 4px;">{analisis['patron_plumas']}</div>
            </div>
            <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 12px;">
                <span style="color: #ff7b72; font-weight: bold;">🎨 Fase Cromática (HSV):</span> 
                <div style="color: #e6edf3; font-size: 14px; margin-top: 4px;">{analisis['mineral_predominante']}</div>
            </div>
            <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 12px;">
                <span style="color: #7ee787; font-weight: bold;">🌊 Evaluación de Orillas:</span> 
                <div style="color: #e6edf3; font-size: 14px; margin-top: 4px;">{analisis['actividad_enzimatica']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_bloque_der:
            st.markdown("<p style='color: #aaa; font-size: 13px; font-weight: bold; margin-bottom: 8px;'>📍 COORDENADAS DE ORIGEN (VISTA DE SATÉLITE)</p>", unsafe_allow_html=True)
            
            col_lat, col_lon = st.columns(2)
            with col_lat:
                lat_input = st.number_input("Latitud", value=-12.0464, format="%.6f", key=f"lat_{st.session_state.id_actual}")
            with col_lon:
                lon_input = st.number_input("Longitud", value=-77.0428, format="%.6f", key=f"lon_{st.session_state.id_actual}")
            
            try:
                import folium
                from streamlit_folium import st_folium
                
                m = folium.Map(location=[lat_input, lon_input], zoom_start=15, tiles=None)
                
                # Servidor directo de capas satelitales de Google Earth
                folium.TileLayer(
                    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                    attr='Google Satellite',
                    name='Google Earth',
                    overlay=False,
                    control=True
                ).add_to(m)
                
                folium.Marker(
                    [lat_input, lon_input],
                    popup=f"Muestra: {st.session_state.id_actual}",
                    icon=folium.Icon(color="green", icon="leaf")
                ).add_to(m)
                
                st_folium(m, height=205, use_container_width=True, key=f"map_{st.session_state.id_actual}")
                
            except ImportError:
                st.caption("💡 Tip: Instala `pip install streamlit-folium folium` para habilitar el mapa interactivo de Google Earth.")
                df_mapa = {"lat": [lat_input], "lon": [lon_input]}
                st.map(df_mapa, height=205)
                
        st.divider()
        
        txt_oxi, txt_min, txt_bio, conclusion = generar_informe(
            st.session_state.id_actual, analisis, datos.get('caract', [])
        )
        
        st.markdown("### Interpretación")
        
        with st.expander("Oxigenación", expanded=False):
            st.info(txt_oxi)
        with st.expander("Mineralización", expanded=False):
            st.info(txt_min)
        with st.expander("Biología", expanded=False):
            st.info(txt_bio)
        
        st.markdown("---")
        st.markdown(conclusion)
        
        with st.expander("Perfil radial", expanded=False):
            perfil = analisis['perfil_radial']
            
            if perfil and len(perfil) > 10:
                indices = list(range(len(perfil)))
                
                fig_perfil = go.Figure()
                fig_perfil.add_trace(go.Scatter(
                    x=indices,
                    y=perfil,
                    mode='lines',
                    line=dict(color='#2ecc71', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(46, 204, 113, 0.15)'
                ))
                
                max_r_pixeles = analisis['max_r']
                paso = analisis['paso']
                
                idx_nucleo_fin = int((max_r_pixeles * 0.12) / paso)
                idx_min_inicio = int((max_r_pixeles * 0.35) / paso)
                idx_min_fin = int((max_r_pixeles * 0.68) / paso)
                idx_bio_inicio = int((max_r_pixeles * 0.76) / paso)
                idx_bio_fin = int((max_r_pixeles * 0.94) / paso)
                
                fig_perfil.add_vrect(x0=0, x1=idx_nucleo_fin, fillcolor="rgba(46, 204, 113, 0.25)", line_width=0)
                fig_perfil.add_vrect(x0=idx_min_inicio, x1=idx_min_fin, fillcolor="rgba(52, 152, 219, 0.25)", line_width=0)
                fig_perfil.add_vrect(x0=idx_bio_inicio, x1=idx_bio_fin, fillcolor="rgba(241, 196, 15, 0.25)", line_width=0)
                
                fig_perfil.update_layout(
                    height=350,
                    xaxis_title="Distancia desde el centro",
                    yaxis_title="Intensidad de gris",
                    template="plotly_dark",
                    showlegend=False
                )
                
                st.plotly_chart(fig_perfil, use_container_width=True)
                st.caption("🟢 Oxigenación | 🔵 Mineralización | 🟡 Biología")
else:
    st.info("Cargue una imagen para comenzar")
