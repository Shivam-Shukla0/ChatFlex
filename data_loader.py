def loadDataset(text, chunk_size=300):
    """Split raw dataset text into smaller chunks."""
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks
