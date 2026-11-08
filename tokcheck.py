from tokenizers import Tokenizer

t = Tokenizer.from_file("tokenizer/tokenizer.json")
s = "如何摆脱拖延症？"
enc = t.encode(s)
print("ids   :", enc.ids)
print("tokens:", enc.tokens)
print("decode:", repr(t.decode(enc.ids)))
print()
print("vocab size:", t.get_vocab_size())
for i in enc.ids:
    print(f"  {i}: {t.id_to_token(i)!r}")
