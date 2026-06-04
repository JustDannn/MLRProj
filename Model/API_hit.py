import pandas as pd
import ollama

# Load data awal lu
df_awal = pd.read_csv('Data/data.csv')
hasil_eksperimen = []

daftar_bahasa = ['inggris', 'jawa', 'sunda']
daftar_model = ['llama3', 'qwen2', 'gemma2'] 

print("Memulai eksperimen MLR jalur Ollama (Lokal)...")

for idx, row in df_awal.iterrows():
    id_kalimat = row['id_kalimat']
    
    for bahasa in daftar_bahasa:
        teks_konten = row[f'teks_{bahasa}']
        panjang_kata = len(str(teks_konten).split())
        prompt_final = f"Continue the following text in the exact same language. Maximum two sentences: {teks_konten}"
        
        for model_name in daftar_model:
            print(f"Menguji Kalimat #{id_kalimat} | Bahasa: {bahasa.upper()} | Model: {model_name}")
            
            try:
                response = ollama.generate(model=model_name, prompt=prompt_final)
                
                t_in = response.get('prompt_eval_count', 0)
                t_out = response.get('eval_count', 0)
                latensi_detik = response.get('eval_duration', 0) / 1e9
                
                hasil_eksperimen.append({
                    'id_kalimat': id_kalimat,
                    'bahasa': bahasa,
                    'model': model_name,
                    'panjang_kata_input': panjang_kata,
                    'token_input': t_in,
                    'token_output': t_out,
                    'latensi': round(latensi_detik, 4)
                })
                
            except Exception as e:
                print(f"Error pada Kalimat #{id_kalimat} [{model_name} - {bahasa}]: {e}")
                continue

# Save datanya
df_final = pd.DataFrame(hasil_eksperimen)
df_final.to_csv('Data/data_withAI.csv', index=False)
print("\nDone!")