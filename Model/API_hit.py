import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
GEMINI_KEY = os.getenv("Gemini")

df_awal = pd.read_csv('Data/data.csv')

hasil_eksperimen = []

def hit_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.5}
    }
    res = requests.post(url, json=payload).json()
    
    if 'error' in res:
        raise Exception(f"Gemini Error: {res['error']['message']}")
        
    usage = res.get('usageMetadata', {})
    if not usage:
         raise Exception(f"Gemini Response Aneh: {res}")
         
    return usage['promptTokenCount'], usage['candidatesTokenCount']

def hit_openrouter(prompt, model_name):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name, 
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }
    res = requests.post(url, headers=headers, json=payload).json()
    
    if 'error' in res:
        raise Exception(f"OpenRouter Error: {res['error']['message']}")
        
    return res['usage']['prompt_tokens'], res['usage']['completion_tokens']

daftar_bahasa = ['inggris', 'jawa', 'sunda']
daftar_model = {
    'Gemini': 'google/gemma-4-31b-it:free',
    'Llama-3': 'meta-llama/llama-3.1-8b-instruct:free',
    'Mistral': 'qwen/qwen-2.5-7b-instruct:free'
}

print("Hitting the LLM API...")

for idx, row in df_awal.iterrows():
    id_kalimat = row['id_kalimat']
    
    for bahasa in daftar_bahasa:
        teks_konten = row[f'teks_{bahasa}']
        panjang_kata = len(teks_konten.split())        
        prompt_final = f"Continue the following text in the exact same language. Maximum two sentences: {teks_konten}"
        
        for nama_model, id_model in daftar_model.items():
            print(f"Menguji Kalimat #{id_kalimat} | Bahasa: {bahasa.upper()} | Model: {nama_model}")
        
            time.sleep(2) 
            
            start_time = time.time()
            try:
                if nama_model == 'Gemini':
                    t_in, t_out = hit_gemini(prompt_final)
                else:
                    t_in, t_out = hit_openrouter(prompt_final, id_model)
                
                latensi = time.time() - start_time
                
                hasil_eksperimen.append({
                    'id_kalimat': id_kalimat,
                    'bahasa': bahasa,
                    'model': nama_model,
                    'panjang_kata_input': panjang_kata,
                    'token_input': t_in,
                    'token_output': t_out,
                    'latensi': round(latensi, 4)
                })
                
            except Exception as e:
                print(f"Error pada Kalimat #{id_kalimat} [{nama_model} - {bahasa}]: {e}")
                continue

# Export data akhir ke CSV
df_final = pd.DataFrame(hasil_eksperimen)
df_final.to_csv('LLM_Result.csv', index=False)
print("\n'LLM_Result.csv' sukses dibuat.")