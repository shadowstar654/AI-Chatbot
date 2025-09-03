import argparse, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--persona", type=str, required=True, help="Persona name (e.g. friend, clown, mentor)")
args = parser.parse_args()

# --- Load dataset file ---
dataset_path = f"datasets/{args.persona}.json"
with open(dataset_path, "r") as f:
    data = json.load(f)

# --- Format dataset ---
# We wrap it in a simple "User: ... PersonaBot: ..." style so the model learns the dialogue pattern.
train_data = [{"text": f"User: {d['input']} {args.persona.capitalize()}Bot: {d['output']}"} for d in data]
dataset = Dataset.from_list(train_data)

# --- Load base model ---
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
# --- Tokenize ---
def tokenize(example):
    #return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    encodings = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings
tokenized = dataset.map(tokenize)

# --- Training config ---
output_dir = f"./{args.persona}bot"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,                  # you can raise this if dataset is larger
    per_device_train_batch_size=2,
    save_total_limit=1,
    logging_steps=5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

# --- Train and Save ---
print(f"Training {args.persona.capitalize()}Bot...")
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir) 
print(f"{args.persona.capitalize()}Bot saved to {output_dir}")

