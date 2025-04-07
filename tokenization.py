import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4o')

print("Vocab Size", encoder.n_vocab) # 2,00,019 (200K)

text = "I can do this all day"
tokens = encoder.encode(text)

print("Tokens", tokens) # Tokens [40, 665, 621, 495, 722, 2163]

my_tokens = [976, 9059, 10139, 402, 290, 2450]
decoded = encoder.decode([976, 9059, 10139, 402, 290, 2450])
print("Decoded", decoded)
