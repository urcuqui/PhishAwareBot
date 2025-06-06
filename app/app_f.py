# app.py

import os
import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------- CONFIGURACIÓN -----------------

MODEL_NAME = "WhiteRabbitNeo/WhiteRabbitNeo-V3-7B"
RUTA_CACHE = "D:/torch"
FEEDBACK_CSV = "./feedback_phishing.csv"  # mismo CSV usado por train.py
PEFT_OUTPUT_DIR = "./fine_tuned_model"    # directorio donde están los pesos LoRA

# 1) Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=RUTA_CACHE
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 2) BitsAndBytesConfig (coincidir con train.py)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 3) Cargar modelo base y aplicar LoRA (pesos finos ya guardados)
@st.cache_resource(show_spinner=False)
def cargar_modelo_lo_ra():
    # 3.1) Cargar modelo base en 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        cache_dir=RUTA_CACHE,
        quantization_config=bnb_config
    )
    base_model = prepare_model_for_kbit_training(base_model)
    # 3.2) LoRA (misma configuración que en train.py)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj',
                        'q_proj', 'o_proj', 'down_proj']
    )
    model = get_peft_model(base_model, peft_config)
    # 3.3) Cargar pesos guardados (si existen)
    if os.path.isdir(PEFT_OUTPUT_DIR):
        model.load_state_dict(
            torch.load(os.path.join(PEFT_OUTPUT_DIR, "pytorch_model.bin")),
            strict=False
        )
    model.eval()
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

model = cargar_modelo_lo_ra()

# 4) Funciones de generación
def generar_texto(prompt: str, max_length: int = 256) -> str:
    """
    Genera texto a partir de un prompt, usando el modelo con LoRA.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        max_new_tokens=max_length,
        pad_token_id=tokenizer.eos_token_id
    )
    texto_generado = tokenizer.decode(out[0], skip_special_tokens=True)
    # El método .decode incluye el prompt; recortamos para devolver solo la parte nueva
    return texto_generado[len(prompt):].strip()

# 5) Inicializar CSV si no existe
if not os.path.exists(FEEDBACK_CSV):
    df_inicial = pd.DataFrame(columns=["prompt", "generated", "label"])
    df_inicial.to_csv(FEEDBACK_CSV, index=False)

# -------------- STREAMLIT --------------

st.set_page_config(page_title="Generador de Emails (Phishing vs Real)", layout="centered")

st.title("🖋️ Generador y Etiquetador de Emails")
st.write("""
Esta app te permite generar ejemplos de **correo de phishing** y **correo real**, 
dar tu feedback y luego esos datos servirán para reentrenar el modelo.  
""")

# 1) Selector de tipo de correo
tipo = st.radio(
    "¿Qué tipo de correo quieres generar?",
    ("Phishing", "Real legítimo")
)

# 2) Prompt por defecto según tipo
if tipo == "Phishing":
    prompt_base = "Human: Crea un correo de phishing en español\nAssistant:"
else:
    prompt_base = "Human: Crea un correo real de bienvenida en español\nAssistant:"

# 3) Botón de generación
if st.button(f"Generar ejemplo de {tipo}"):
    with st.spinner("Generando texto…"):
        ejemplo = generar_texto(prompt_base)
    st.code(ejemplo, language="plaintext")
    # Mostramos el área de feedback
    st.write("**¿Es este ejemplo un correo de** ", tipo, "**?**")
    label = st.radio(
        "Selecciona:", 
        ("Sí, coincide con el tipo", "No, no coincide")
    )
    etiqueta_final = tipo.lower() if label.startswith("Sí") else ("real" if tipo == "Phishing" else "phishing")

    if st.button("Enviar feedback"):
        # Guardar en CSV: prompt, generated, label (“phishing” o “real”)
        df_feedback = pd.read_csv(FEEDBACK_CSV)
        nueva_fila = {
            "prompt": prompt_base,
            "generated": ejemplo,
            "label": etiqueta_final
        }
        df_feedback = pd.concat([df_feedback, pd.DataFrame([nueva_fila])], ignore_index=True)
        df_feedback.to_csv(FEEDBACK_CSV, index=False)
        st.success("👍 Gracias por tu feedback. Se ha guardado correctamente.")

# 4) Mostrar resumen de feedback recolectado
st.markdown("---")
st.subheader("📊 Resumen de feedback recolectado")
df_feedback = pd.read_csv(FEEDBACK_CSV)
st.write(f"Total de ejemplos guardados: **{len(df_feedback)}**")
if len(df_feedback) > 0:
    # Mostrar las primeras filas
    st.dataframe(df_feedback.tail(10))

st.markdown("---")
st.write("""
> Cuando tengas suficientes ejemplos (por ejemplo, al menos 10 phishing y 10 reales), 
> abre la terminal y ejecuta:
>
> ```bash
> python train.py
> ```
> 
> para reentrenar el modelo con los ejemplos etiquetados.
""")
