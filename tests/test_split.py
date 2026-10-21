from src.ingestion.chunker import split_long_text, count_tokens

# Make an artificially long piece of text to force splitting
long_text = ("This is a test sentence about company policy. " * 60)
print("Total tokens in long_text:", count_tokens(long_text))

pieces = split_long_text(long_text, max_tokens=50, overlap_tokens=10)
print(f"\nSplit into {len(pieces)} pieces\n")
for i, p in enumerate(pieces):
    print(f"--- Piece {i+1} ({count_tokens(p)} tokens) ---")
    print(p[:120], "...")