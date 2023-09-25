import nltk
# nltk.download()
# load all items from NLTK book module
from nltk.book import *

print(text1)
print(text2)

# concordance -> every occurance of a given word
text1.concordance("monstrous")
text2.concordance("affection")

# similar -> the relevant word in parentheses
text1.similar("monstrous")

# common _contexts: contexts that are shared by two or more words
text2.common_contexts(["monstrous", "very"])

# dispersion plot: locations and frequencies of occurrences
# -> showing future's positional information
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

# generate some random text
text3.generate()

# lexical diversity of text3
len(set(text3)) / len(text3)

# frequency distribution: frequency of each vocabulary item in the text
fdist1 = FreqDist(text1)
fdist1.most_common(50)

# focusing on long word; should be more characteristics and informative
long_words = [w for w in set(text1) if len(w) > 15]
sorted(long_words)

# collection: a sequence of words that occur together unusually often
list(bigrams(['more', 'is', 'said', 'than', 'done']))
