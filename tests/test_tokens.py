from src.ingestion.chunker import count_tokens

print(count_tokens("How many sick days do I get?"))
print(count_tokens("This is a much longer sentence that should have a noticeably higher token count than the short one above it."))