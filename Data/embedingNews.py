import os
import requests
import pandas as pd
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

load_dotenv()
ACCESS_KEY = os.getenv("newsAPI")
offset=0
url = "http://api.mediastack.com/v1/news"
kalimat_inggris = []

while len(kalimat_inggris) < 50:
    print(f"Mencari data... (Offset: {offset}, Terkumpul: {len(kalimat_inggris)}/50)")
    
    params = {
        'access_key': ACCESS_KEY,
        'languages': 'en',
        'categories': '-general,-sports',
        'sort': 'published_desc',
        'limit': 100,
        'offset': offset 
    }

    response = requests.get(url, params=params)
    data = response.json()
    
    if 'data' not in data or not data['data']:
        print("Data habis atau ada error dari API.")
        break

    for artikel in data['data']:
        deskripsi = artikel.get('description')
        
        if deskripsi and isinstance(deskripsi, str):
            jumlah_kata = len(deskripsi.split())
            if 10 <= jumlah_kata <= 45:
                deskripsi_bersih = deskripsi.replace('\n', ' ').replace('\r', '').strip()
                
                if deskripsi_bersih not in kalimat_inggris:
                    kalimat_inggris.append(deskripsi_bersih)
                    
        if len(kalimat_inggris) == 50:
            break
            
    offset += 100
print(f"Dapat {len(kalimat_inggris)} deskripsi berita bahasa Inggris yang valid!")

print("Mulai proses auto-translate ke Jawa dan Sunda...")
dataset = []
translator_jv = GoogleTranslator(source='en', target='jw')
translator_su = GoogleTranslator(source='en', target='su')

for i, teks in enumerate(kalimat_inggris):
    print(f"Translating baris ke-{i+1}...")
    teks_jv = translator_jv.translate(teks)
    teks_su = translator_su.translate(teks)
    
    dataset.append({
        'id_kalimat': i + 1,
        'teks_inggris': teks,
        'teks_jawa': teks_jv,
        'teks_sunda': teks_su,
        'panjang_kata_inggris': len(teks.split())
    })

df = pd.DataFrame(dataset)
df.to_csv('Data/data.csv', index=False)
print("Selesai! File 'data.csv' siap dipakai buat eksperimen LLM.")