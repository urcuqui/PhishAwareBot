# PhishAwareBot

PhishAwareBot is a Streamlit-based web application that generates and displays two types of emails: one phishing and one legitimate (benign), both in Spanish. The user is challenged to identify which email is phishing and which is benign, providing feedback to help raise awareness about phishing tactics.


## Demo

See PhishAwareBot in action:  
<img src="demo4.gif" alt="PhishAwareBot Demo" width="600"/>


## Features

- **AI-Generated Emails:** Uses a TinyLlama-based language model to generate realistic phishing and benign emails in Spanish.
- **Interactive Interface:** Users can generate new email pairs and submit their guesses about which is phishing and which is benign.
- **Immediate Feedback:** The app provides instant feedback on the user's choices, helping to educate and improve phishing detection skills.

## How It Works

1. The app loads a TinyLlama language model with PPO (Proximal Policy Optimization) reinforcement learning setup.
2. When the user clicks "Generar Correos", the app generates one phishing and one benign email.
3. The user reviews both emails and selects which is phishing and which is benign.
4. The app checks the answers and displays whether the user was correct.

## Requirements

- Python 3.13+
- See `requirements.txt` for all dependencies.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/PhishAwareBot.git
   cd PhishAwareBot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

## File Structure

- `app.py`: Main Streamlit application.
- `requirements.txt`: List of required Python packages.

## License

This project is for educational and research purposes only.

---

*Created by Christian Urcuqui.*