import sys
import os
import torch

# 1. Path Fix: Help Python find main.py in the parent folder
# This is crucial for your MLOps pipeline to work on Render
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from main import MiniGPT, chars, encode 
except ImportError:
    print("❌ Error: Could not find main.py. Check your folder structure.")
    sys.exit(1)

# 2. Load the 29,643 bytes of data you just verified
input_path = os.path.join(os.path.dirname(__file__), '..', 'input.txt')
with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 3. Training Setup
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
vocab_size = len(chars) 

# Initialize model with the NEW vocab size (42)
model = MiniGPT(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 4. Training Loop
print(f"🚀 Starting training with vocab size: {vocab_size}")
model.train()
for iter in range(500):
    # Standard Mini-GPT batching logic
    ix = torch.randint(len(train_data) - 8, (32,))
    x = torch.stack([train_data[i:i+8] for i in ix])
    y = torch.stack([train_data[i+1:i+8+1] for i in ix])
    
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % 100 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")

# 5. Save the NEW model.pt to the backend folder
save_path = os.path.join(os.path.dirname(__file__), '..', 'model.pt')
torch.save(model.state_dict(), save_path)
print(f"✅ Success! New model.pt saved at: {save_path}")