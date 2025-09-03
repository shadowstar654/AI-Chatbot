from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Personas available ---
modes = {
    "friend": "./friendbot",   # trained model folder
    #"clown": "./clownbot"      # not trained yet, placeholder
}

print("Choose chatbot mode:")
for key in modes:
    print("-", key)

choice = input("Enter mode: ").strip().lower()

if choice not in modes:
    print("Mode not available, defaulting to 'friend'")
    choice = "friend"

model_path = modes[choice]
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(f"Loaded {choice} mode!")
bot_name = f"{choice.capitalize()}"
# --- Chat loop ---
while True:
    user = input("You: ")
    if user.lower() == "quit":
        break
    prompt = f"User: {user}\n{choice.capitalize()}Bot:"
    response = chat(prompt, max_new_tokens=150, top_p=0.9, top_k=50, temperature=0.7, repetition_penalty=1.2, truncation=True, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    #print(response.split(f"{choice.capitalize()}Bot:")[-1].strip())
    bot_response = response.split(f"{bot_name}Bot:")[-1].strip()
    print(f"{bot_name}: {bot_response}")
