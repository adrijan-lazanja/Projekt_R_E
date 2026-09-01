import os
from dotenv import load_dotenv
load_dotenv()
import json
import csv
import requests

from datetime import datetime
from constants import (
    ServiceType, OPENAI_CHAT_MODEL, OPENAI_CHAT_TEMPERATURE,
    RETRIEVER_SCORE_THRESHOLD, RETRIEVER_MAX_RETURNED, SYSTEM_PROMPT_TEMPLATE
)

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    print("GREŠKA: Nedostaje biblioteka. Instaliraj: pip install openai")
    exit(1)


IOT_LOGS_PATH = "logs.csv"

class AgronomistAgent:
    def __init__(self, knowledge_base_path="usjevi.json"):
        self.kb_path = knowledge_base_path
        self.knowledge_base = self._load_data()
        self.chat_history = []

    def _log(self, service: ServiceType, message):
        """Pomoćna funkcija za ispis logova u konzoli"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{service.name}] {message}")

    def _load_data(self):
        """Učitava JSON bazu"""
        if not os.path.exists(self.kb_path): 
            print(f"Upozorenje: Datoteka {self.kb_path} ne postoji!")
            return []
        with open(self.kb_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
            print(f"[SYSTEM] Učitano {len(data)} kultura iz baze.")
            return data

    def _get_latest_iot_data(self):
        """Računa PROSJEK vlage tla i rada pumpe iz logs.csv"""
        if not os.path.exists(IOT_LOGS_PATH):
            return "Nema dostupnih IoT mjerenja (logs.csv nedostaje)."

        try:
            with open(IOT_LOGS_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if not rows:
                    return "Tablica mjerenja je prazna."

                count = len(rows)
                sum_soil = 0.0
                sum_pump = 0.0

                for row in rows:
                    try:
                        sum_soil += float(row.get('soil_moisture', 0))
                        sum_pump += float(row.get('pump_output', 0))
                    except ValueError:
                        continue

                avg_report = f"""
PODACI SENZORA (Na temelju {count} mjerenja):
- Vlaga tla: {sum_soil / count:.1f} %
- Rad pumpe: {sum_pump / count:.1f} %
"""
                return avg_report
        except Exception as e:
            return f"Greška pri čitanju senzora: {str(e)}"
            


    def _get_weather_data(self):
        """Dohvaća trenutnu vremensku prognozu i prognozu za idućih 24h s OpenWeatherMap API-ja"""
        api_key = os.getenv("OPENWEATHERMAP_API_KEY", "")
        city = os.getenv("OPENWEATHERMAP_CITY", "Zagreb")

        if not api_key:
            return "Vremenska prognoza nije dostupna (OPENWEATHERMAP_API_KEY nije postavljen)."

        try:
            current_url = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?q={city}&appid={api_key}&units=metric&lang=hr"
            )
            current = requests.get(current_url, timeout=5)
            current.raise_for_status()
            cur = current.json()

            forecast_url = (
                f"https://api.openweathermap.org/data/2.5/forecast"
                f"?q={city}&appid={api_key}&units=metric&lang=hr&cnt=8"
            )
            forecast = requests.get(forecast_url, timeout=5)
            forecast.raise_for_status()
            fc = forecast.json()

            temp       = cur['main']['temp']
            feels_like = cur['main']['feels_like']
            humidity   = cur['main']['humidity']
            description= cur['weather'][0]['description']
            wind_speed = cur['wind']['speed']
            rain_1h    = cur.get('rain', {}).get('1h', 0)

            weather_str = f"""
TRENUTNA VREMENSKA PROGNOZA ({city}):
- Temperatura: {temp:.1f} °C (osjet: {feels_like:.1f} °C)
- Opis: {description}
- Vlaga zraka: {humidity} %
- Brzina vjetra: {wind_speed:.1f} m/s
- Kiša (zadnji sat): {rain_1h:.1f} mm

PROGNOZA ZA IDUĆIH 24 SATA:
"""
            for item in fc.get('list', []):
                t    = item['main']['temp']
                desc = item['weather'][0]['description']
                rain = item.get('rain', {}).get('3h', 0)
                weather_str += f"- {item['dt_txt']}: {t:.1f}°C, {desc}, kiša: {rain:.1f} mm\n"

            self._log(ServiceType.MICROSERVICE, f"Vremenska prognoza za '{city}' uspješno dohvaćena.")
            return weather_str

        except Exception as e:
            return f"Greška pri dohvaćanju vremenske prognoze: {str(e)}"

    def _guardrail_check(self, question):
        """GPT klasifikator koji provjerava je li pitanje vezano uz poljoprivredu"""
        self._log(ServiceType.MICROSERVICE, "Guardrail: provjera relevantnosti pitanja...")
        prompt = (
            "Ti si dio agronomskog savjetničkog sustava za hrvatsko podneblje. "
            "Tvoj jedini zadatak je klasificirati je li korisnikovo pitanje relevantno za agronomskog asistenta.\n"
            "Relevantna su pitanja vezana uz: uzgoj bilja, kulture (pšenica, kukuruz, voće, povrće itd.), "
            "tlo, navodnjavanje, sadnju, berbu, pesticide, gnojiva, meteorologiju, vremensku prognozu, "
            "senzore na polju (vlaga tla, temperatura, pumpa), agronomsku statistiku, "
            "ili bilo kakvo savjetovanje o tome što i kako uzgajati.\n"
            "Odgovori SAMO jednom riječju: DA ili NE.\n\n"
            f"Pitanje: {question}"
        )
        try:
            response = client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )
            result = response.choices[0].message.content.strip().upper()
            self._log(ServiceType.MICROSERVICE, f"Guardrail rezultat: {result}")
            return result.startswith("DA")
        except Exception as e:
            self._log(ServiceType.MICROSERVICE, f"Guardrail greška: {e} — propuštam pitanje")
            return True

    def _apply_filters(self, items, category_filter=None):
        """Simulira GraphRAG filtriranje (ArangoDB logika)"""
        if not category_filter:
            return items
        
        filtered = [i for i in items if i.get("kategorija", "").lower() == category_filter.lower()]
        self._log(ServiceType.RETRIEVER, f"Filter '{category_filter}': {len(items)} -> {len(filtered)} dokumenata")
        return filtered

    def _format_statistics(self, stats):
        """Pretvara složeni JSON objekt statistike u čitljiv tekst za AI"""
        if isinstance(stats, dict):
            lines = []
            for k, v in stats.items():
                clean_key = k.replace("_", " ").capitalize()
                lines.append(f"{clean_key}: {v}")
            return "; ".join(lines)
        return str(stats)

    
    def _retriever_service(self, query, category_filter=None):
        """Naprednija logika pretrage otporna na gramatiku"""
        self._log(ServiceType.RETRIEVER, f"Tražim: '{query}' (Filter: {category_filter})")
        
        # Prvo filtriram po kategoriji
        candidates = self._apply_filters(self.knowledge_base, category_filter)
        
        hits = []
        query_terms = query.lower().split()
        
        for item in candidates:
            item_text = json.dumps(item, ensure_ascii=False).lower()
            score = 0
            
            # Ako odgovara kategoriji, daj mu male početne bodove
            if category_filter:
                score = 1.0 
            
            for term in query_terms:
                if len(term) < 3: continue    # Da preskoči "i", "u", "je"
                
                # Uzimam samo prva 4 slova riječi
                root = term[:4]
                
                if root in item_text:
                    score += 0.5
            
            # Dodatni bodovi za točno ime kulture
            if item.get("kultura", "").lower() in query.lower():
                score += 2.0 

            if score >= 0.5:
                hits.append({"doc": item, "score": score})
        
        return hits


    def process_request(self, user_question):
        if not self._guardrail_check(user_question):
            return ("Specijaliziran sam isključivo za agronomska pitanja. "
                    "Molim Vas postavite pitanje vezano uz uzgoj kultura, tlo, "
                    "navodnjavanje, klimatske uvjete ili srodnu tematiku.")

        cat_filter = None
        q_lower = user_question.lower()
        if any(x in q_lower for x in ["voće", "voćk", "jabuk", "krušk", "maslin", "grožđ", "mandarin"]):
            cat_filter = "Voće"
        elif any(x in q_lower for x in ["žit", "pšenic", "kukuruz", "ječam"]):
            cat_filter = "Žitarice"
        elif any(x in q_lower for x in ["povr", "rajčic", "krastav", "lubenic"]):
            cat_filter = "Povrće"

        # Retrieve
        raw_hits = self._retriever_service(user_question, category_filter=cat_filter)
        
        # Rerank
        raw_hits.sort(key=lambda x: x["score"], reverse=True)
        best_docs = [h['doc'] for h in raw_hits[:RETRIEVER_MAX_RETURNED]]
        
        # Formatiranje DZS konteksta
        context_str = ""
        if not best_docs:
            context_str = "Nema specifičnih podataka u DZS bazi za ovaj upit."
        else:
            for doc in best_docs:
                stats_readable = self._format_statistics(doc.get('dzs_statistika', {}))
                
                context_str += f"""
--- PREDMET: {doc.get('kultura')} ({doc.get('kategorija')})
 REGIJE: {', '.join(doc.get('regija_preporuka', []))}
 STATITISTIKA (DZS 2024): {stats_readable}
 UVJETI UZGOJA: {doc.get('uvjeti_uzgoja')}
"""

        # Učitavam IoT mjerenja i vremensku prognozu
        iot_data_str     = self._get_latest_iot_data()
        weather_data_str = self._get_weather_data()

        # Spajanje DZS, IoT i vremenskog konteksta u jedan prompt
        full_context = f"""
BAZA ZNANJA (DZS):
{context_str}

{iot_data_str}

{weather_data_str}
"""

        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history[-2:]])

        # LLM generiranje
        final_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            context=full_context, 
            history=history_str,
            question=user_question
        )
        
        self._log(ServiceType.LLM, "Generiranje odgovora...")
        try:
            response = client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=OPENAI_CHAT_TEMPERATURE
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Greška u komunikaciji s AI servisom: {e}\n(Provjeri API ključ)"
        
        self.chat_history.append({"role": "user", "content": user_question})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        return answer


if __name__ == "__main__":
    agent = AgronomistAgent()
    print("\n🌱 AgroGenie (DZS Edition) je spreman!")
    print("Probajte pitati: 'Kakvo je stanje s mandarinama?' ili 'Koliko pšenice je proizvedeno?'\n")
    
    while True:
        q = input("Pitanje (q za izlaz): ")
        if q.lower() in ['q', 'exit']: break
        resp = agent.process_request(q)
        print(f"\nAGENT: {resp}\n")
        print("-" * 50)