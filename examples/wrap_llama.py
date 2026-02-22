from transformers import pipeline
from cfi.core import CFI

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")  # Replace with your model

cfi = CFI()
H_t = []
S_prev = cfi.get_initial_state()

prompt = input("User prompt: ")
u_t = prompt
band, guidance, S_new = cfi(u_t, H_t, S_prev)

if band == 'block':
    print("Blocked: High risk")
elif band == 'steer':
    prompt += " " + guidance['rephrase_suggestion']
response = pipe(prompt, max_length=100)[0]['generated_text']
print(response)

H_t.append(u_t + " " + response)
S_prev = S_new
