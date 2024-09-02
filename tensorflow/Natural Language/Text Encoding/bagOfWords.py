vocab = {} #We inicialize an empty dictionary to save our words
word_encoding = 1 #We start encoding words at 1

def bagOfWords(sentence):
    global word_encoding

    words = sentence.lower().split(" ") #We split the sentence into words
    bag = {} #We inicialize an empty dictionary to save our bag of words

    for word in words:
        if word in vocab:
            encoding = vocab[word]  #Get encoding from vocab
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1
        
        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1

    return bag

test_text = "this is a test to see if this test will work is is test a a"
bag = bagOfWords(test_text)
print(bag)
print(vocab)