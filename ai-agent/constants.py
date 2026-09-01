import os
from enum import Enum

class ServiceType(Enum):
    MICROSERVICE = 0
    MEGASERVICE = 1
    EMBEDDING = 1
    RETRIEVER = 2
    RERANK = 3
    LLM = 4

class ServiceRoleType(Enum):
    MICROSERVICE = 0
    MEGASERVICE = 1

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
OPENAI_CHAT_TEMPERATURE = float(os.getenv("OPENAI_CHAT_TEMPERATURE", 0.01))

RETRIEVER_SCORE_THRESHOLD = float(os.getenv("RETRIEVER_ARANGO_SCORE_THRESHOLD", 0.5))
RETRIEVER_MAX_RETURNED = int(os.getenv("RETRIEVER_ARANGO_TRAVERSAL_MAX_RETURNED", 3))

SYSTEM_PROMPT_TEMPLATE = """
### Uloga:
Ti si stručan agronomski asistent za hrvatsko podneblje.
Koristi priložene "Rezultate pretrage" (iz lokalne baze DZS RH) kao primarni izvor istine.

### Upute za formatiranje:
Odgovore uvijek oblikuj koristeći Markdown sintaksu:
- Koristi **podebljani tekst** za ključne pojmove i brojke.
- Koristi nenumeriranu listu (- stavka) za nabrajanje preporuka ili činjenica.
- Koristi numeriranu listu za korake ili prioritete.
- Koristi kratke naslove (## ili ###) ako odgovor ima više tematskih cjelina.
- Nikada ne vraćaj odgovor kao jedan dugačak paragraf bez strukture.

### Rezultati pretrage (Kontekst):
{context}

### Povijest:
{history}

### Pitanje:
{question}
"""
