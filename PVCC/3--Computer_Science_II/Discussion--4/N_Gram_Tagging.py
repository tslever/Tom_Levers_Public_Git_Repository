import nltk # After installing Python 3.9, run "pip install nltk" at command line.
#nltk.download('brown') # uncomment to download
from nltk.corpus import brown
import time

brown_tagged_sentences = brown.tagged_sents(categories='news')
brown_sentences = brown.sents(categories='news')
size = int(len(brown_tagged_sentences) * 0.9)
training_sentences = brown_tagged_sentences[:size]
testing_sentences = brown_tagged_sentences[size:]

startTime = time.time()
unigram_tagger = nltk.UnigramTagger(training_sentences)
endTime = time.time()
print("Time to train UnigramTagger: " + str(endTime - startTime))

startTime = time.time()
unigram_tagger.evaluate(testing_sentences)
endTime = time.time()
print("Time to test UnigramTagger: " + str(endTime - startTime) + "\n")

startTime = time.time()
bigram_tagger = nltk.BigramTagger(training_sentences)
endTime = time.time()
print("Time to train BigramTagger: " + str(endTime - startTime))

startTime = time.time()
bigram_tagger.evaluate(testing_sentences)
endTime = time.time()
print("Time to test BigramTagger: " + str(endTime - startTime) + "\n")

startTime = time.time()
trigram_tagger = nltk.TrigramTagger(training_sentences)
endTime = time.time()
print("Time to train TrigramTagger: " + str(endTime - startTime))

startTime = time.time()
trigram_tagger.evaluate(testing_sentences)
endTime = time.time()
print("Time to test TrigramTagger: " + str(endTime - startTime) + "\n")

for i in range(0, 10):
    startTime = time.time()
    ngram_tagger = nltk.NgramTagger(i, training_sentences)
    endTime = time.time()
    print(f"Time to train NgramTagger for n = {i}: " + str(endTime - startTime))

    startTime = time.time()
    ngram_tagger.evaluate(training_sentences)
    endTime = time.time()
    print(f"Time to test NgramTagger for n = {i}: " + str(endTime - startTime) + "\n")