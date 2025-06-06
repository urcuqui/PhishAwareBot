# train.py

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
import pandas as pd

# -------------- CONFIGURACIÓN BÁSICA --------------

# Ajusta estas rutas según tu sistema:
RUTA_CACHE = "D:/torch"
MODEL_NAME = "WhiteRabbitNeo/WhiteRabbitNeo-V3-7B"
PEFT_OUTPUT_DIR = "./fine_tuned_model"  # donde guardaremos LoRA + pesos afinados
FEEDBACK_CSV = "./feedback_phishing.csv"  # archivo donde Streamlit guardará el feedback

# 1) Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    cache_dir=RUTA_CACHE
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 3) Configuración de bits-and-bytes para cargar en 4-bit (opcional)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 4) LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj',
                    'q_proj', 'o_proj', 'down_proj']
)

# 5) Cargar modelo principal en 4-bit + prepararlo para k-bit training
print("⏳ Cargando modelo base y preparándolo para LoRA...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    cache_dir=RUTA_CACHE,
    quantization_config=bnb_config
)
base_model = prepare_model_for_kbit_training(base_model)  # habilita gradient checkpointing, etc.

# 6) Cargar modelo de referencia (full-precision; usará para DPO)
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    cache_dir=RUTA_CACHE
)

# 7) Aplicar LoRA sobre el modelo base
model = get_peft_model(base_model, peft_config)

# 8) Configuración de entrenamiento DPO
dpo_training_args = DPOConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=10,              # ajústalo más grande si quieres más epochs
    save_strategy="no",
    logging_steps=1,
    output_dir=PEFT_OUTPUT_DIR,
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True,
    report_to=None  # o "wandb" si tienes Weights & Biases configurado
)

def cargar_feedback_como_dataset(csv_path: str) -> Dataset:
    """
    Lee el archivo CSV con los ejemplos anotados por el usuario
    en Streamlit y lo convierte en un HuggingFace Dataset
    con el formato que espera DPOTrainer: 
      cada fila debe tener dos columnas 'chosen' y 'rejected'.

    Supondremos que el CSV tiene estas columnas:
      - prompt: el prompt original usado para generación
      - generated: el texto completo generado por el modelo
      - label: "phishing" o "real" (basado en feedback)
    
    Para entrenamiento con DPO, necesitamos pares (chosen,rejected). 
    Una posible forma de armarlos:
      - Si label == "phishing", lo ponemos como 'chosen'
        y buscamos un ejemplo 'real' al azar en el mismo CSV como 'rejected' (y viceversa).
      - Eso crea pares de preferencia “prefiero phishing sobre real” 
        y “prefiero real sobre phishing” (o podrías filtrar unidireccionalmente).

    Este ejemplo ilustra la creación de pares simples:
    """
    df = pd.read_csv(csv_path)
    # Filtrar ejemplos etiquetados
    phishing_df = df[df["label"] == "phishing"].reset_index(drop=True)
    real_df = df[df["label"] == "real"].reset_index(drop=True)

    pares = []
    # Tomaremos min(len(phishing), len(real)) y armar pares
    n_pares = min(len(phishing_df), len(real_df))
    for i in range(n_pares):
        phishing_text = phishing_df.loc[i, "generated"]
        real_text     = real_df.loc[i, "generated"]

        # Par: el phishing es 'chosen', el real es 'rejected'
        pares.append({
            "chosen": f"Human: Genera un correo de phishing\nAssistant: {phishing_text}",
            "rejected": f"Human: Genera un correo real de bienvenida\nAssistant: {real_text}"
        })
        # (Opcional) También puedes invertir la preferencia en otro par:
        pares.append({
            "chosen": f"Human: Genera un correo real de bienvenida\nAssistant: {real_text}",
            "rejected": f"Human: Genera un correo de phishing\nAssistant: {phishing_text}"
        })

    # Convertir a Dataset de HF
    if len(pares) == 0:
        raise ValueError(f"No hay suficientes ejemplos en {csv_path} para crear pares.")
    return Dataset.from_list(pares)

def entrenar_con_feedback(csv_path: str):
    """
    Lee el CSV de feedback, arma un Dataset y entrena (o refina) el modelo.
    """
    print("📥 Leyendo feedback del usuario y creando dataset DPO...")
    dataset = cargar_feedback_como_dataset(csv_path)

    print("🚀 Iniciando entrenamiento DPO con los pares generados...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        processing_class=tokenizer  # igual que en tu ejemplo
    )
    dpo_trainer.train()
    print("✅ Fine-tuning completado. Guardando pesos LoRA...")
    model.save_pretrained(PEFT_OUTPUT_DIR)
    print(f"✅ Modelo guardado en: {PEFT_OUTPUT_DIR}")