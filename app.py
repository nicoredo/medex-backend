from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import httpx
import re
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Podés usar ["https://medex.ar"] en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/evaluar_ia")
async def evaluar_ia(request: Request):
    try:
        data = await request.json()
        print("✅ JSON recibido:", data)

        texto_hc = data.get("datos", {}).get("texto_hc", "")

        with open("criterios_estudios_textual.json", "r", encoding="utf-8") as f:
            criterios = json.load(f)["estudios"]
            print("📂 Archivo de criterios cargado exitosamente")

        prompt = armar_prompt(texto_hc, criterios)
        print("📤 Enviando prompt estructurado a OpenRouter...")

        respuesta_ia = await consultar_openrouter(prompt)
        print("✅ Respuesta JSON recibida")

        # Regex más tolerante (ya no exige ```json ... ```)
        json_match = re.search(r"\[.*?\]", respuesta_ia, re.DOTALL)
        if not json_match:
            raise ValueError("No se encontró un bloque de array JSON válido.")

        json_parsed = json.loads(json_match.group(0))

        # Agregar descripciones desde archivo local
        descripciones = {e['nombre']: e['descripcion'] for e in criterios}
        for est in json_parsed:
            est['descripcion'] = descripciones.get(est['nombre'], "Sin descripción")

        return {"estudios": json_parsed}

    except Exception as e:
        print("❌ ERROR EN /evaluar_ia:", e)
        return {"error": str(e)}


@app.get("/criterios")
def obtener_criterios():
    try:
        with open("criterios_estudios_textual.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"estudios": data["estudios"]}
    except Exception as e:
        print("❌ ERROR EN /criterios:", e)
        return {"error": str(e)}


def armar_prompt(texto_hc, estudios):
    def formatear_criterios(est):
        criterios = est["criterios"]
        texto = "Criterios de inclusión:\n"
        for c in criterios.get("inclusion", []):
            if c["tipo"] == "grupo-condicional":
                texto += f"- {c['condicion']}:\n"
                for opc in c["opciones"]:
                    texto += f"    • {opc['condicion']}\n"
            else:
                texto += f"- {c['condicion']}\n"

        if criterios.get("exclusion"):
            texto += "\nCriterios de exclusión:\n"
            for c in criterios["exclusion"]:
                texto += f"- {c['condicion']}\n"
        return texto.strip()

    criterios_txt = "\n\n".join(
        f"{i+1}. {e['nombre']}: {e['descripcion']}\n{formatear_criterios(e)}"
        for i, e in enumerate(estudios)
    )

    prompt_final = f"""
Sos un evaluador de criterios de seleccion de pacientes para estudios de investigacion, interpreta el caso presentado.

### Instrucciones:
- Analizá los criterios de inclusión de los estudios (Excluido = si tiene algun criterio de exclusion; Cumple totalmente = todos in excepcion deben cumplirse; Cumple parcial = le falta un criterio para cumplir totalmente).
- Los criterios pueden incluir antecedentes, medicacion. Valores de laboratorio tiene que estrictamente estar en los rangos de los criterios, en caso que no comprendas el valor o las unidades, pedile ampliar con informacion adicional.
- Respondé solamente si el paciente cumple totalmente (✅) o parcialmente (⚠️). No incluyas estudios no aplicables (❌).
- Si un dato no está mencionado en el texto clínico, asumí que NO está presente, es estricto sin alucinaciones.
- Devolvé un array JSON con los siguientes campos por estudio:

```json
[
  {{
    "nombre": "Nombre del estudio",
    "estado": "✅ / ⚠️",    
    "detalle": "Motivo del cumplimiento parcial o total"
  }}
]

---

Texto clínico del paciente:
{texto_hc}

---

Estudios clínicos disponibles:
{criterios_txt}
""".strip()

    return prompt_final


async def consultar_openrouter(prompt):
    try:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not OPENROUTER_API_KEY:
            raise ValueError("Falta la API KEY de OpenRouter en variables de entorno.")

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://medex.ar",
            "Content-Type": "application/json"
        }

        body = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 3000
        }

        async with httpx.AsyncClient() as client:
            r = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
            print("📨 STATUS:", r.status_code)
            print("📨 TEXT:", r.text)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print("❌ ERROR EN consultar_openrouter:", e)
        raise


from fastapi import UploadFile, File
from docx import Document

@app.post("/subir_word")
async def subir_word(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.docx", "wb") as f:
        f.write(contents)

    doc = Document("temp.docx")
    texto = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return {"texto": texto}

