from transformers import AutoTokenizer, AutoModelForCausalLM

# Upravte cestu k modelu, pokud jste ho stáhli do jiné složky
model_path = "./models/Llama-3-13b-chat"

# Načtení tokenizéru a modelu s povoleným důvěřováním vzdálenému kódu
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

# Vstupní prompt - můžete zadat libovolnou otázku či zprávu
prompt = "Ahoj, jak se máš?"

# Příprava vstupu
inputs = tokenizer(prompt, return_tensors="pt")

# Generování odpovědi s maximálním počtem nových tokenů
outputs = model.generate(**inputs, max_new_tokens=100)

# Výpis vygenerované odpovědi
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
