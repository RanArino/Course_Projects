from nltk.corpus import wordnet as wn

# define a list of pairs
pairs = [
    ("car", "automobile"), ("gem", "jewel"), ("journey", "voyage"),
    ("boy", "lad"), ("coast", "shore"), ("asylum", "madhouse"),
    ("magician", "wizard"), ("midday", "noon"), ("furnace", "stove"),
    ("food", "fruit"), ("bird", "cock"), ("bird", "crane"),
    ("tool", "implement"), ("brother", "monk"), ("lad", "brother"),
    ("crane", "implement"), ("journey", "car"), ("monk", "oracle"),
    ("cemetery", "woodland"), ("food", "rooster"), ("coast", "hill"),
    ("forest", "graveyard"), ("shore", "woodland"), ("monk", "slave"),
    ("coast", "forest"), ("lad", "wizard"), ("chord", "smile"),
    ("glass", "magician"), ("rooster", "voyage"), ("noon", "string")
]

# define function to rank the similarity score
def get_similarity_scores(pairs: list[tuple]):
    """
    Ranks the pairs in order of decreasing similarity.
    """
    # define a ditionary of each pair score
    scores_dict = {}

    # traversing all pairs
    for w1, w2 in pairs:
        # define synsets of each word
        syn1_sets = wn.synsets(w1)
        syn2_sets = wn.synsets(w2)

        # get the maximum score between the pairs of sybsets
        max_score = 0
        # traversing all syn1 sets and syn2 sets
        for syn1 in syn1_sets:
            for syn2 in syn2_sets:
                # get similality score
                score = syn1.path_similarity(syn2) # type: ignore
                if max_score == 0 or max_score < score:
                    max_score = score
        # assign pair names and its similality score
        scores_dict[(w1, w2)] = max_score

    # return score dictionary after sorting based on its value
    return sorted(scores_dict.items(), key=lambda d: d[1], reverse=True)

# run function
sim_scores = get_similarity_scores(pairs)
print('Similarity Scores: ')
print(sim_scores)