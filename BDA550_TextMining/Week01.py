#####   Class   #####
"""

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

"""


#####   Workshop   #####
from nltk.book import FreqDist
from nltk.corpus import names
male_names = names.words('male.txt')
female_names = names.words('female.txt')

# Check name list for both male and female
#print("Lists of Names: ")
#print("Males: ", male_names[:10])
#print("Females: ", female_names[:10])
#print('')

# Create lists of initial latter for both male and female
male_initL = [name[0] for name in male_names]
female_initL = [name[0] for name in female_names]
#print("Lists of Initial Latters: ")
#print("Males: ", male_initL[:10])
#print("Females: ", female_initL[:10])
#print('')

# Define frequent distribution for each gender's initial latter
fdist_male = FreqDist(male_initL)
fdist_female = FreqDist(female_initL)

# Show the most frequent initial latter for each gender
print('')
print("Question 01:")
print("Most frequent initial latter: ")
# Print the table
print(f'{"":<10}{"":<10}{"Latter":<10}{"Count":<10}')
print(f'{"Male":<10}{"":<10}{fdist_male.most_common(1)[0][0]:<10}{fdist_male.most_common(1)[0][1]:<10}')
print(f'{"Female":<10}{"":<10}{fdist_female.most_common(1)[0][0]:<10}{fdist_female.most_common(1)[0][1]:<10}')
print('')


