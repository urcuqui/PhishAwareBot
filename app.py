import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset as HFDataset

# 1) Reward model dummy
class DummyRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, **kwargs):
        if "input_ids" in kwargs and isinstance(kwargs["input_ids"], torch.Tensor):
            bsz = kwargs["input_ids"].shape[0]
            return torch.zeros((bsz, 1), device=kwargs["input_ids"].device)
        return torch.zeros((0, 1))

@st.cache(allow_output_mutation=True)
def load_model():
    # Tokenizer y modelos
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLMWithValueHead.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Inyectar generation_config y base_model_prefix
    model.generation_config = model.config
    ref_model.generation_config = ref_model.config
    model.base_model_prefix = "transformer"
    ref_model.base_model_prefix = "transformer"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ref_model.to(device)

    # Configuración PPO
    config = PPOConfig(
        exp_name="ppo_example",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        learning_rate=1.41e-5,
        batch_size=1,
    )

    # Dataset dummy
    dummy_data = {"query": [""]}
    dummy_dataset = HFDataset.from_dict(dummy_data)

    # Reward model y optimizador
    reward_model = DummyRewardModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Instanciar PPOTrainer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model,
        tokenizer,
        dummy_dataset,
        optimizer
    )

    return tokenizer, ppo_trainer, device

def generate_email(prompt, tokenizer, ppo_trainer, device):
    # Tokenizar el prompt
    tokenized = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids = tokenized["input_ids"].to(device)

    # Convertir a lista de tensores 1D
    query_list = [input_ids[0]]

    # Generar con PPOTrainer
    response_ids = ppo_trainer.generate(
        query_list,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
    )
    # Decodificar
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)    
    return response.strip()

# Cargar modelo una sola vez
tokenizer, ppo_trainer, device = load_model()

st.title("Generador de Correos: Phishing vs Benignos")

# Botón para generar correos
if st.button("Generar Correos"):
    with st.spinner("Generando correos..."):
        phishing_prompt = "Crea un correo phishing en español."
        benign_prompt = "make a legitimate and formal email in Spanish."

        phishing_email = generate_email(phishing_prompt, tokenizer, ppo_trainer, device)        
        benign_email = generate_email(benign_prompt, tokenizer, ppo_trainer, device)

        # Guardar en session_state
        st.session_state["emails"] = {
            "A": {"text": phishing_email, "label": "Phishing"},
            "B": {"text": benign_email, "label": "Benign"}
        }

if "emails" in st.session_state:
    emails = st.session_state["emails"]
    st.subheader("Email A")
    st.text_area("Contenido de Email A", value=emails["A"]["text"], height=200)
    st.subheader("Email B")
    st.text_area("Contenido de Email B", value=emails["B"]["text"], height=200)

    # Formulario de feedback
    st.subheader("Identifica cuál es phishing y cuál es benigno")
    label_A = st.radio("¿Email A es phishing o benigno?", ("Phishing", "Benign"), key="label_A")
    label_B = st.radio("¿Email B es phishing o benigno?", ("Phishing", "Benign"), key="label_B")

    if st.button("Enviar Feedback"):
        correct_A = (label_A == emails["A"]["label"])
        correct_B = (label_B == emails["B"]["label"])

        if correct_A and correct_B:
            st.success("¡Correcto! Identificaste ambos correos correctamente.")
        else:
            st.error("Hay errores en tu identificación.")
            st.write(f"Email A correcto: {emails['A']['label']}")
            st.write(f"Email B correcto: {emails['B']['label']}")
