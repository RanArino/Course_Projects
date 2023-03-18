"""
File: anagram_analysis.py

Name: Ran Arino
ID: 153073200


Two classes for creating binary trees
Example:
    intial_tree = BinaryTree.THE_EMPTY_TREE
    first_word = BinaryTree("a")


1st func: anagram_finder(file)
    Takes a direct file path and returns the anagram table (pd.DataFrame).
    Each row of its table has unique set of words and frequency in a given file.

2nd func: anagram_search(given_word, anagram_df)
    Takes a given word and anagram_df from the function "anagram_finder"
    If a word is found, return all anagrams and its frequency, 1st and 2nd elements of result, respectively. 
    If a word is not found, the function returns "Not found".

"""

# load packages 
import pandas as pd
import nltk
from nltk.corpus import stopwords

class EmptyTree(object):
    """Represents an empty tree."""

    def isEmpty(self):
        return True

    def __str__(self):
        return ""

    def __iter__(self):
        """Iterator for the tree."""
        return iter([])

    def preorder(self, lyst):
        return

    def inorder(self, lyst):
        return

    def postorder(self, lyst):
        return
class BinaryTree(object):
    """Represents a nonemoty binary tree."""

    # Singleton for all empty tree objects
    THE_EMPTY_TREE = EmptyTree()

    def __init__(self, item):
        """Creates a tree with
        the given item at the root."""
        self._root = item
        self._left = BinaryTree.THE_EMPTY_TREE
        self._right = BinaryTree.THE_EMPTY_TREE

    def isEmpty(self):
        return False

    def getRoot(self):
        return self._root

    def getLeft(self):
        #Returns the left subtree. Precondition: T is not an empty tree.
        return self._left
    
    def getRight(self):
        #Returns the right subtree. Precondition: T is not an empty tree.
        return self._right

    def setRoot(self, item):
        self._root = item

    def setLeft(self, tree):
        #Sets the left subtree to tree. Precondition: T is not an empty tree.
        self._left = tree
    
    def setRight(self, tree):
        #Sets the right subtree to tree. Precondition: T is not an empty tree.
        self._right = tree

    def removeLeft(self):
        """Removes and returns the left subtree.
        Precondition: T is not an empty tree. Postcondition: the left subtree is empty."""
        left = self._left
        self._left = BinaryTree.THE_EMPTY_TREE
        return left
    
    def removeRight(self):
        """Removes and returns the right subtree.
        Precondition: T is not an empty tree. Postcondition: the right subtree is empty."""
        right = self._right
        self._right = BinaryTree.THE_EMPTY_TREE
        return right

    def __str__(self):
        """Returns a string representation of the tree
        rotated 90 degrees to the left."""
        def strHelper(tree, level):
            result = ""
            if not tree.isEmpty():
                result += strHelper(tree.getRight(), level + 1)
                result += "| " * level
                result += str(tree.getRoot()) + "\n"
                result += strHelper(tree.getLeft(), level + 1)
            return result
        return strHelper(self, 0)

    def __iter__(self):
        """Iterator for the tree."""
        lyst = []
        self.inorder(lyst)
        return iter(lyst)

    def inorder(self, lyst):
        """Adds items to lyst during
        an inorder traversal."""
        self.getLeft().inorder(lyst)
        lyst.append(self.getRoot())
        self.getRight().inorder(lyst)
  
    def height(self):
        """
        Set left to point to the left subtree
        Set right to point to the right subtree
        You may use getLeft() or getRight()

        If you have reached the leaf, return 0.
        Otherwise, recursively add 1 to the maximum hight.
        """
        left = self.getLeft()
        right = self.getRight()
        if left.isEmpty() and right.isEmpty():
            return 0
        else:
            return 1 + max(left.height(), right.height())

    def leaves(self):
        """
        Set left to point to the left subtree
        Set right to point to the right subtree
        You may use getLeft() or getRight()

        If you have reached the leaf, return it (call getRoot).
        Otherwise, recursively add the leaves on the left subtree and the right subtree
        """
        left = self.getLeft()
        right = self.getRight()
        if self.isEmpty():
            return []
        elif left.isEmpty() and right.isEmpty():
            return [self.getRoot()]
        else:
            return left.leaves() + right.leaves()
        
    def add(self, newItem):
        """
        Adds newItem to the tree if it's not already in it
        or replaces the existing item if it is in it.
        Returns None if the item is added or the old
        item if it is replaced.
        """
        # Helper function to search for item's position 
        def addHelper(self):
            currentItem = self.getRoot()
            left = self.getLeft()
            right = self.getRight()
            # New item is less, go left until spot is found
            if newItem < currentItem:
                if left.isEmpty():
                    self.setLeft(BinaryTree(newItem))
                else:
                    addHelper(left)
            # New item is greater or equal, 
            # go right until spot is found
            elif right.isEmpty():
                self.setRight(BinaryTree(newItem))
            else:
                addHelper(right)
            # End of addHelper
        # Tree is empty, so new item goes at the root
        if self.isEmpty():
            self = BinaryTree(newItem)
        # Otherwise, search for the item's spot
        else:
            addHelper(self)

            
            
def anagram_finder(file_name):
    """
    Load text file and return the anagram table (pd.DataFrame).
    Each row of its table has unique set of words and frequency.
    """
    anagram_df = pd.DataFrame(columns=['words', 'freq'])
    freq_dict = {}
    word = ''
    
    # list of stop words
    stop_words = stopwords.words('english')
    stop_words
    
    # load the whole text file 
    with open(file_name, 'r') as f:
        data = f.read()

    for i in range(len(data)):
        """ 
        Traversing every character
        Time Complexity = O(n1), where n1 is number of characters in 'data'
        """    
        # a string (lowered) is an alphabet... 
        if data[i].isalpha() == True:
            # if data[i] is 1st charactor...
            if i == 0:
                # create new binary tree
                tree = BinaryTree(data[i].lower())

            # if data[i-1] is not an alphabet...
            elif data[i-1].isalpha() != True:
                # if word (cumulative characters until non-alphabet reaches) is not stop words...
                if word not in stop_words:
                    """
                    Traversing every stop words (Once computer faces True, skip all codes inside this if statement)
                    Time Complexity(worst) = O(n2), where n2 is len(stop_words), which is 179.
                    """
                    
                    # running in-order traversal to change alphabetical order word by word
                    inorder_list = []
                    tree.inorder(inorder_list)
                    """
                    Traversing every character of each word (every item in a tree)
                    Time Complexity = O(n3), where n3 is len(word)
                    """

                    # the result of inorder as a dictionay key
                    key = f"{inorder_list}"
                    # if the value of a dictionary key already exists...
                    if freq_dict.get(key) != None:
                        # 1st element: words of anagram (no duplication)
                        freq_dict[key]["words"].add(word)
                        """
                        Adding item in a set
                        Time Complexity = O(1) in average.
                        Although the worst case is O(n5), where n4 is the length of each set, 
                        considering the small length of the set, no huge impact would affect on the operations.
                        """
                        
                        # 2nd element: freqency of anagrams 
                        freq_dict[key]["freq"] += 1
                        # creating anagram_df if the length of set is more than 2.
                        if len(freq_dict[key]["words"]) >= 2:
                            anagram_df.loc[key] = freq_dict[key]

                    else:
                        freq_dict[key] = {"words": {word}, "freq": 1}

                # reset/recreate tree and word
                word = ''
                tree = BinaryTree(data[i].lower())

            else:
                # adding new item to the existing tree
                tree.add(data[i].lower())

            # adding character to "word"    
            word += data[i].lower()
    
    # anagram table
    return anagram_df



def anagram_search(given_word: str, anagram_df: pd.DataFrame):
    """
    Takes a given word and anagram_df from the function "anagram_finder"
    Returns two elements: 1st is all anagrams, and 2nd is its frequency
    If a given word is not found, the function returns "Not found".
    """
    # create new tree with the 1st item as a root
    tree = BinaryTree(given_word[0].lower())
    
    # traversing the rest of items
    for i in given_word[1:]:
        """
        Traversing every character.
        Time Conplexity = O(n), where n is the number of characters of a given word
        """
        tree.add(i.lower())
        
    inorder_list = []
    tree.inorder(inorder_list)
    """
    Traversing every items in tree.
    Time Complexity = O(n), where n is the number of characters of a given word
    """
    
    try:
        anagram =  anagram_df.loc[str(inorder_list)]['words']
        freq =  anagram_df.loc[str(inorder_list)]['freq']
        return (anagram, freq)
    except:
        return 'Not Found'

    
    
def main():
    # the file must be in the same location
    file_name = "adventures_of_huckleberry_finn.txt"
    # create anagram dataframe
    anagram_df = anagram_finder(file_name)
    # enter a given word as an input
    given_word = str(input("Enter a word here: "))
    # start searching for a given word
    result = anagram_search(given_word, anagram_df)
    # showing the rsult
    if len(result) == 2:
        return print(f"Anagrams: {result[0]}, Frequency: {result[1]}")
    else:
        return print("Not Found")

if __name__ == "__main__": main()