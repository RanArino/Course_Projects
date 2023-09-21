import nltk
from nltk.book import FreqDist
from nltk.corpus import names, stopwords

import random
from typing import Optional

male_names = names.words('male.txt')
female_names = names.words('female.txt')

# Create lists of initial latter for both male and female
male_initL = [name[0] for name in male_names]
female_initL = [name[0] for name in female_names]

# Define frequent distribution for each gender's initial latter
fdist_male = FreqDist(male_initL)
fdist_female = FreqDist(female_initL)

# Show the most frequent initial latter for each gender
print('')
print("=== Question 01 ===")
print('')
# Print the table
# Male
print(f'{"Male":<10}{"Latter":<10}{"Count":<10}')
for i, (w, f) in enumerate(fdist_male.most_common(5)):
    print(f'{i+1:<10}{"":<3}{w:<10}{f:<10}')
# Female
print('')
print(f'{"Female":<10}{"Latter":<10}{"Count":<10}')
for i, (w, f) in enumerate(fdist_female.most_common(5)):
    print(f'{i+1:<10}{w:<10}{f:<10}')


print("")
print("")
print("=== Qiestion 02 ===")
# get stopwords list
stop_words = stopwords.words('english')
# open text file
with open('BDA550_TextMining/warlordofmars.txt', 'r') as f:
    text = f.read()

# create a list of words
words_list = [w for w in text.split()]

# Define useful functions
def filter_stopwords(words_list: list, stop_words: list):
    """
    Return list of words after removing stopwords
    """
    return [w for w in words_list if not w.lower() in stop_words]

# (a): Function: return 50 most freq words (no stopwords)
def freq_words_top50(words_list: list, stop_words: list):
    """
    Return 50 most frequently occuring words words without stopwords.
    """
    # return filtered text
    words_list = filter_stopwords(words_list, stop_words)
    # create freqency distribution
    fd = FreqDist(words_list)
    # get only words
    fd_words = list(fd.keys())

    return fd_words[:50]

# showing result
print('')
print('2-(a): 50 Most Freq Words')
print(freq_words_top50(words_list, stop_words))


# (b): Function: return 50 most freq bigrams
def freq_bigram_top50(words_list: list, stop_words: list):
    """
    Return the 50 most frequent bigrams
    """
    # return filtered text
    words_list = filter_stopwords(words_list, stop_words)
    # get list of bigrams
    bigrams_list = nltk.bigrams(words_list)
    # get freq dist
    fd = FreqDist(bigrams_list)
    # get list of bigram pairs
    fd_bigrams = list(fd.keys())

    return fd_bigrams[:50]

# showing result
print('')
print('2-(b): 50 Most Freq Bigrams')
print(freq_bigram_top50(words_list, stop_words))


# (c): get a randomly selected word from the top n freqency distribution
def random_word_n_freq(n: int, words_list: list, stop_words: Optional[list] = None):
    """
    Return a randomly selected word from the 'n' most likely words.
    """
    # remove stopwords if users assign a list of stopwords
    if stop_words:
        words_list = filter_stopwords(words_list, stop_words)

    # get frequency distribution
    fd = FreqDist(words_list)
    # get list of the n most likely words
    fd_top_n = list(fd.keys())[:n]
    # get a randomly selected word
    rand_word = random.choice(fd_top_n)

    return rand_word

print('')
n = random.randint(1, 100)  # define 'n' randomly
print(f'2-(c): A randomly selected word from the {n} most likely words.')
print(random_word_n_freq(n, words_list, stop_words))
    

# (d): Train a model on this corpus and get i to generate random text
#  define function to generate model
def generate_model(cfd, word, num=15):
    seq_words = ''
    for i in range(num):
        seq_words += word + " "
        word = cfd[word].max()

    return seq_words
#  create bigrams list
bigrams = nltk.bigrams(words_list)
#  create conditional drequency distribution
cfd = nltk.ConditionalFreqDist(bigrams)
#  define a list of start words, which are randomly selected from the 100 most freq words.
num_start_words = 10
start_words_list = [random_word_n_freq(100, words_list, stop_words) for _ in range(num_start_words)]
print('')
print('2-(d): Generate random texts')
# generate random texts 
for start in start_words_list:
    print(generate_model(cfd, start))

# Discussions
print('')
print(
"""
Strengths:
- Since the next word will be selected based on the highest likelihood of the word following a given word,
   the part of the generated text can have a simple language structure.

Weaknesses:
- The root of the issue is that the determination of the next word sticks to the previous word, 
   thereby a certain start word always specifies the same word for its next;
   for example, the next word of "I" is always "had', and it will continue "been", "a", "moment", and so on.
- It means that the likelihood of the next words cannot consider the sentence overall. 
- Also, it implies that as the number of generated words text ("num" argument of "generate_model") is increasing,
   the result is likely to close to the same sentence regardless of what a start word is.
"""
)
print('')

