vocab = {} 
word_encoding = 1

def one_hot_encoding(text):
    global word_encoding

    words = text.lower().split(' ')
    encoding = []

    for word in words:

        if word in vocab:
            code = vocab[word]
            encoding.append(code)
        else:
            vocab[word] = word_encoding
            encoding.append(word_encoding)
            word_encoding += 1

    return encoding

test_text = "this is a test to see if this test will work is is test a a"
encoding = one_hot_encoding(test_text)
print(encoding)
print(vocab)
    