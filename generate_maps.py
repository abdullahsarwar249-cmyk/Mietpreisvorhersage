"""
Vereinfachte interaktive Kartenerstellung 
Erstellt eigenständiges HTML mit Markern und Heatmaps
"""

import pickle
import json

print("="*80)
print("INTERAKTIVE KARTEN ERSTELLEN ")
print("="*80)

# Ergebnisse laden
with open('evaluation_results.pkl', 'rb') as f:
    eval_results = pickle.load(f)

test_df = eval_results['test_data']

# Regional coordinates (German cities)
region_coords = {
    'Baden_Württemberg': [48.7758, 9.1829],
    'Bayern': [48.7758, 11.4328],
    'Berlin': [52.5200, 13.4050],
    'Brandenburg': [52.1189, 12.6084],
    'Bremen': [53.1045, 8.8017],
    'Hamburg': [53.5511, 9.9937],
    'Hessen': [50.1109, 8.6821],
    'Mecklenburg_Vorpommern': [53.6154, 12.9289],
    'Niedersachsen': [52.6362, 9.7753],
    'Nordrhein_Westfalen': [51.4556, 7.0116],
    'Rheinland_Pfalz': [49.9929, 6.9623],
    'Saarland': [49.2557, 6.9983],
    'Sachsen': [51.0834, 13.5325],
    'Sachsen_Anhalt': [51.6921, 11.8797],
    'Schleswig_Holstein': [54.5973, 9.8773],
    'Thüringen': [50.9840, 11.0290]
}

# Aggregate data by region
print("[1] Bereite regionale Daten vor...")
regional_stats = test_df.groupby('regio1').agg({
    'baseRent': ['mean', 'std', 'min', 'max', 'count'],
    'prediction_xgb': 'mean',
    'livingSpace': 'mean',
    'noRooms': 'mean'
}).round(2)

regional_stats.columns = ['_'.join(col).strip() for col in regional_stats.columns.values]
regional_stats = regional_stats.reset_index()

# Rename aggregated columns for easier access
regional_stats.rename(columns={
    'baseRent_mean': 'baseRent_mean',
    'prediction_xgb_mean': 'prediction_xgb',
    'livingSpace_mean': 'livingSpace_mean',
    'noRooms_mean': 'noRooms_mean'
}, inplace=True)

# Map numeric regio1 IDs to region names
region_name_map = {
    0: 'Baden_Württemberg',
    1: 'Bayern',
    2: 'Berlin',
    3: 'Brandenburg',
    4: 'Bremen',
    5: 'Hamburg',
    6: 'Hessen',
    7: 'Mecklenburg_Vorpommern',
    8: 'Niedersachsen',
    9: 'Nordrhein_Westfalen',
    10: 'Rheinland_Pfalz',
    11: 'Saarland',
    12: 'Sachsen',
    13: 'Sachsen_Anhalt',
    14: 'Schleswig_Holstein',
    15: 'Thüringen'  # If 15th region
}

regional_stats['region_name'] = regional_stats['regio1'].map(region_name_map)

print(f"   Found {len(regional_stats)} regions")

# ============================================================================
# MIETPREISKARTE ERSTELLEN
# ============================================================================
print("\n[2] Erstelle interaktive Mietpreiskarte als HTML...")

min_rent = regional_stats['baseRent_mean'].min()
max_rent = regional_stats['baseRent_mean'].max()

def get_color(rent):
    """Farbe basierend auf Mietwert zurückgeben"""
    normalized = (rent - min_rent) / (max_rent - min_rent)
    if normalized < 0.2:
        return '#2ecc71'  # Green
    elif normalized < 0.4:
        return '#f1c40f'  # Yellow
    elif normalized < 0.6:
        return '#e67e22'  # Orange
    elif normalized < 0.8:
        return '#e74c3c'  # Red
    else:
        return '#c0392b'  # Dark Red

# Markerdaten erstellen
markers_data = []
for idx, row in regional_stats.iterrows():
    region_name = row['region_name']
    if region_name in region_coords:
        lat, lon = region_coords[region_name]
        avg_rent = row['baseRent_mean']
        
        markers_data.append({
            'lat': lat,
            'lon': lon,
            'title': region_name,
            'rent': avg_rent,
            'color': get_color(avg_rent),
            'radius': 8 + (avg_rent - min_rent) / (max_rent - min_rent) * 12,
            'count': int(row['baseRent_count']),
            'std': row['baseRent_std'],
            'min': row['baseRent_min'],
            'max': row['baseRent_max'],
            'living_space': row['livingSpace_mean'],
            'rooms': row['noRooms_mean']
        })

# HTML-Vorlage
html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Mietpreisvorhersage - Interaktive Mietpreiskarte</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        html, body {
            height: 100%;
            font-family: 'Arial', sans-serif;
        }
        
        #map {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 100%;
            height: 100%;
            background: #f0f0f0;
        }
        
        .info-box {
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 300px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            font-size: 13px;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .info-box h3 {
            margin: 0 0 10px 0;
            font-size: 16px;
            border-bottom: 2px solid #333;
            padding-bottom: 8px;
        }
        
        .color-legend {
            margin: 15px 0;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 10px;
            border: 1px solid #333;
        }
        
        .popup-title {
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 8px;
            color: #333;
        }
        
        .popup-content {
            font-size: 12px;
            line-height: 1.6;
        }
        
        .popup-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        
        .popup-label {
            font-weight: bold;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="info-box">
        <h3> Regionale Mietpreisübersicht</h3>
        <p style="margin-bottom: 10px; color: #666;">
         Diese Karte zeigt die durchschnittlichen Nettokaltmieten der analysierten Regionen in Deutschland. Klicken Sie auf einen Kreis, um weitere Informationen zur jeweiligen Region anzuzeigen.
        </p>
        
        <div class="color-legend">
            <strong>Preiskategorien:</strong>
            <div class="legend-item">
                <div class="legend-color" style="background: #2ecc71;"></div>
                <span>Sehr erschwinglich (€{{MIN_RENT}}-€550)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #f1c40f;"></div>
                <span>Erschwinglich (€550-€700)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e67e22;"></div>
                <span>Mittel (€700-€850)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e74c3c;"></div>
                <span>Teuer (€850-€1050)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #c0392b;"></div>
                <span>Sehr hochpreisig (ab €1050)</span>
            </div>
        </div>
        
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
        
        <div style="font-size: 12px;">
            <strong>So verwenden Sie die Karte:</strong><br>
            • Kreisdurchmesser = Mietniveau<br>
            • Klicken Sie auf Kreise für Details<br>
            • Hinein-/Herauszoomen zum Erkunden<br><br>
            
            <strong>Statistiken:</strong><br>
            • Durchschnittliche Nettokaltmiete: €{{AVG_RENT}}
            • Spanne: €{{MIN_RENT}} - €{{MAX_RENT}}<br>
            • Regionen: {{NUM_REGIONS}}
        </div>
    </div>
    
    <script>
        // Initialize map
        const map = L.map('map').setView([51.1657, 10.4515], 6);
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19,
            minZoom: 3
        }).addTo(map);
        
        // Add markers
        const markers = {{MARKERS_JSON}};
        
        markers.forEach(function(marker) {
            const circle = L.circleMarker([marker.lat, marker.lon], {
                radius: marker.radius,
                fillColor: marker.color,
                color: marker.color,
                weight: 2,
                opacity: 0.8,
                fillOpacity: 0.7
            }).addTo(map);
            
            // Popup content
            const popupHtml = `
                <div class="popup-title">${marker.title}</div>
                <div class="popup-content">
                    <div class="popup-row">
                        <span class="popup-label">Durchschnittliche Nettokaltmiete::</span>
                        <span>€${marker.rent.toFixed(2)}/month</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Preisspanne:</span>
                        <span>€${marker.min.toFixed(0)} - €${marker.max.toFixed(0)}</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Std.-Abw.:</span>
                        <span>€${marker.std.toFixed(2)}</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Angebote:</span>
                        <span>${marker.count}</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Durchschnittliche m²:</span>
                        <span>${marker.living_space.toFixed(0)} m²</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Durchschnittliche Zimmer:</span>
                        <span>${marker.rooms.toFixed(1)}</span>
                    </div>
                </div>
            `;
            
            circle.bindPopup(popupHtml);
        });
    </script>
</body>
</html>'''

# Render template using .replace() instead of .format() to avoid conflicts with JS/CSS braces
html_output = html_content
html_output = html_output.replace('{{MARKERS_JSON}}', json.dumps(markers_data))
html_output = html_output.replace('{{MIN_RENT}}', f'{min_rent:.0f}')
html_output = html_output.replace('{{MAX_RENT}}', f'{max_rent:.0f}')
html_output = html_output.replace('{{AVG_RENT}}', f'{regional_stats["baseRent_mean"].mean():.0f}')
html_output = html_output.replace('{{NUM_REGIONS}}', str(len(regional_stats)))

# Karte speichern
with open('interactive_rental_map.html', 'w', encoding='utf-8') as f:
    f.write(html_output)

print("   ✓ Gespeichert: interactive_rental_map.html")

# ============================================================================
# GENAUIGKEITSKARTE ERSTELLEN
# ============================================================================
print("\n[3] Erstelle Genauigkeitskarte als HTML...")

# Calculate errors
regional_stats['prediction_error'] = abs(
    regional_stats['baseRent_mean'] - regional_stats['prediction_xgb']
)

def get_error_color(error):
    """Farbe basierend auf dem Vorhersagefehler"""
    if error < 5:
        return '#2ecc71'
    elif error < 15:
        return '#f1c40f'
    elif error < 30:
        return '#e67e22'
    elif error < 50:
        return '#e74c3c'
    else:
        return '#c0392b'

# Markerdaten erstellen for accuracy
accuracy_markers = []
for idx, row in regional_stats.iterrows():
    region_name = row['region_name']
    if region_name in region_coords:
        lat, lon = region_coords[region_name]
        error = row['prediction_error']
        
        accuracy_markers.append({
            'lat': lat,
            'lon': lon,
            'title': region_name,
            'error': error,
            'color': get_error_color(error),
            'radius': 5 + min(error / 10, 10),
            'actual': row['baseRent_mean'],
            'predicted': row['prediction_xgb'],
            'count': int(row['baseRent_count']),
            'error_pct': (error / row['baseRent_mean'] * 100)
        })

# HTML-Vorlage for accuracy map
accuracy_html = '''<!DOCTYPE html>
<html>
<head>
    <title>Mietpreisvorhersage - Genauigkeitskarte</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        html, body {
            height: 100%;
            font-family: 'Arial', sans-serif;
        }
        
        #map {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 100%;
            height: 100%;
            background: #f0f0f0;
        }
        
        .info-box {
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 320px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            font-size: 13px;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .info-box h3 {
            margin: 0 0 10px 0;
            font-size: 16px;
            border-bottom: 2px solid #333;
            padding-bottom: 8px;
        }
        
        .color-legend {
            margin: 15px 0;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 10px;
            border: 1px solid #333;
        }
        
        .popup-title {
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 8px;
            color: #333;
        }
        
        .popup-content {
            font-size: 12px;
            line-height: 1.6;
        }
        
        .popup-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        
        .popup-label {
            font-weight: bold;
            margin-right: 10px;
        }
        
        .accuracy-good {
            color: green;
            font-weight: bold;
        }
        
        .accuracy-poor {
            color: red;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="info-box">
        <h3>🎯 Genauigkeitskarte</h3>
        <p style="margin-bottom: 10px; color: #666;">
            Modellvorhersagefehler nach Region. Klicken Sie auf Kreise für Details.
        </p>
        
        <div class="color-legend">
            <strong>Accuracy Levels:</strong>
            <div class="legend-item">
                <div class="legend-color" style="background: #2ecc71;"></div>
                <span>Very Accurate (&lt;€5)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #f1c40f;"></div>
                <span>Accurate (&lt;€15)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e67e22;"></div>
                <span>Mittel (&lt;€30)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e74c3c;"></div>
                <span>Inaccurate (&lt;€50)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #c0392b;"></div>
                <span>Poor (&gt;€50)</span>
            </div>
        </div>
        
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
        
        <div style="font-size: 12px;">
            <strong>So verwenden Sie die Karte:</strong><br>
            • Kreisgröße = Höhe des Vorhersagefehlers<br>
            • Klicken Sie auf Kreise für Details<br>
            • Hinein-/Herauszoomen zum Erkunden<br><br>
            
            <strong>Statistiken:</strong><br>
            • Durchschnittlicher Fehler: €{{MEAN_ERROR}}<br>
            • Höchste Genauigkeit: Grün markierte Regionen<br>
            • Analysierte Regionen: {{NUM_REGIONS_ACC}}
        </div>
    </div>
    
    <script>
        // Initialize map
        const map = L.map('map').setView([51.1657, 10.4515], 6);
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19,
            minZoom: 3
        }).addTo(map);
        
        // Add markers
        const markers = {{ACCURACY_JSON}};
        
        markers.forEach(function(marker) {
            const circle = L.circleMarker([marker.lat, marker.lon], {
                radius: marker.radius,
                fillColor: marker.color,
                color: marker.color,
                weight: 2,
                opacity: 0.8,
                fillOpacity: 0.7
            }).addTo(map);
            
            // Popup content
            const accuracyClass = marker.error < 20 ? 'accuracy-good' : 'accuracy-poor';
            const popupHtml = `
                <div class="popup-title">${marker.title}</div>
                <div class="popup-content">
                    <div class="popup-row">
                        <span class="popup-label">Tatsächlicher Durchschnitt:</span>
                        <span>€${marker.actual.toFixed(2)}</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Vorhergesagter Durchschnitt:</span>
                        <span>€${marker.predicted.toFixed(2)}</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Error:</span>
                        <span class="${accuracyClass}">€${marker.error.toFixed(2)} (${marker.error_pct.toFixed(1)}%)</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Angebote:</span>
                        <span>${marker.count}</span>
                    </div>
                </div>
            `;
            
            circle.bindPopup(popupHtml);
        });
    </script>
</body>
</html>'''

# Render accuracy template using .replace() instead of .format()
accuracy_output = accuracy_html
accuracy_output = accuracy_output.replace('{{ACCURACY_JSON}}', json.dumps(accuracy_markers))
accuracy_output = accuracy_output.replace('{{MEAN_ERROR}}', f'{regional_stats["prediction_error"].mean():.2f}')
accuracy_output = accuracy_output.replace('{{NUM_REGIONS_ACC}}', str(len(regional_stats)))

# Genauigkeitskarte speichern
with open('prediction_accuracy_map.html', 'w', encoding='utf-8') as f:
    f.write(accuracy_output)

print("   ✓ Gespeichert: prediction_accuracy_map.html")

print("\n" + "="*80)
print("✅ KARTEN ERFOLGREICH ERSTELLT!")
print("="*80)

print("\n📍 Erzeugte Dateien:")
print("  ✓ interactive_rental_map.html - Mietpreise nach Regionen")
print("  ✓ prediction_accuracy_map.html - Modellgenauigkeit nach Regionen")

print("\n🌐 Karten anzeigen:")
print("  1. HTML-Datei per Doppelklick im Browser öffnen")
print("  2. Oder Rechtsklick → Öffnen mit → Webbrowser")
print("  3. Alternativ die VS-Code-Erweiterung 'Live Server' verwenden")
print("     (empfohlen bei Darstellungsproblemen)")
