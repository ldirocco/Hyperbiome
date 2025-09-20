def recall_at_k(relevant, predicted, k):
    """
    relevant: set o lista di elementi rilevanti
    predicted: lista ordinata dei risultati restituiti dal modello
    k: numero di posizioni da considerare
    """
    relevant = set(relevant)
    top_k = predicted[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant) if relevant else 0.0

# Esempio
relevant = ["A", "B", "C"]
predicted = ["D", "B", "E", "A", "F"]

print(recall_at_k(relevant, predicted, 3))  # 0.333...
print(recall_at_k(relevant, predicted, 5))