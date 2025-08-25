from tqdm import tqdm

def train_model(model, dataloader, optimizer, proxy_loss_fn, device):
    model.train()
    total_loss = 0  # Inizializza la variabile per accumulare la loss
    for x, y in tqdm(dataloader, desc="Train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        emb= model(x)
        loss = proxy_loss_fn(emb, y)  # usa proxy_loss_fn, non emb_loss_fn (che non è definito)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
