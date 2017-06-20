#!/usr/bin/python
# __________________________________________________________________________________________________
# Reads in a Posts.xml. Applies preprocessing (lemmatization...)
#

from lxml import etree
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string
from bs4 import BeautifulSoup
from scoop import futures
import gc


class Loader(object):
    def __init__(self, limit=None, pre_process=True):
        """
        Loader applies pre-processing to a Posts.xml file
        
        :param limit: Maximum number of posts to process
        :param pre_process: Do a pre-processing
        """
        self.posts = []
        self.used_wordnet_tags = {wordnet.NOUN, wordnet.VERB, wordnet.ADV, wordnet.ADJ}
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation_mapping = {key:'' for key in string.punctuation}
        self.limit = limit
        self.do_pre_process = pre_process

    def get_wordnet_tag(self, tag):
        first_character = tag[0]
        if first_character == 'J':
            return wordnet.ADJ
        elif first_character == 'V':
            return wordnet.VERB
        elif first_character == 'N':
            return wordnet.NOUN
        elif first_character == 'R':
            return wordnet.ADV
        else:
            return None

    def pre_process(self, text):
        words = nltk.word_tokenize(text.lower())
        tagged_words = nltk.pos_tag(words)

        words_processed = []
        for tagged_word in tagged_words:
            word = tagged_word[0].translate(self.punctuation_mapping)
            wordnet_tag = self.get_wordnet_tag(tagged_word[1])

            if wordnet_tag in self.used_wordnet_tags:
                word = self.lemmatizer.lemmatize(
                    word, pos=self.get_wordnet_tag(tagged_word[1]))
                if len(word) > 0:
                    words_processed.append(word)

        return words_processed

    def process_row(self, row):
        body = row['Body']
        body = BeautifulSoup(body, 'html.parser').text
        score = int(row['Score'])

        title = []
        if 'Title' in row:
            title = row['Title']
            if self.do_pre_process:
                title = self.pre_process(title)
            else:
                title = title.split()
        tags = []
        if 'Tags' in row:
            tags = row['Tags']
            tags = tags.replace('<', ' ').replace('>', ' ').strip()
            tags = tags.split()

        # pre_process
        if self.do_pre_process:
            body = self.pre_process(body)
        else:
            body = body.split()

        return title, body, tags, score

    def load(self, context, callback=None):
        """
        loads the xml file.
        
        :param context: lxml.etree.iterparse that resembles the row tags
        :param callback: Is called with the batch of posts pre-processed. 
                         If given, posts are not accumulated (memory efficient)
        :return: 
        """
        n_batch = 1000
        
        counter = 0
        to_iterate = []

        while self.limit is None or counter * n_batch < self.limit:
            print('*', end='', flush=True)
            try:
                for i in range(n_batch):
                    _, row = context.__next__()
                    to_iterate.append(dict(row.attrib.items()))
                    row.clear()
                    for ancestor in row.xpath('ancestor-or-self::*'):
                        while ancestor.getprevious() is not None:
                            del ancestor.getparent()[0]
            except StopIteration:
                break
            finally:
                future = futures.map(self.process_row, to_iterate)
                if callback is None:
                    self.posts.extend(list(future))
                else:
                    callback(future)
                to_iterate.clear()
                gc.collect()
            counter += 1
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Read in a stackexchange Posts.xml file (with preprocessing)')
    parser.add_argument('path', type=str, help='path to Posts.xml')
    parser.add_argument('--nopreproc', dest='prepro', default=True, action='store_false', help='no preprocessing')
    parser.add_argument('--limit', default=None, metavar='N', type=int, help='limit read in')
    args = parser.parse_args()

    context = etree.iterparse(args.path, tag='row')
    loader = Loader(limit=args.limit, pre_process=args.prepro)
    loader.load(context)

    for row in loader.posts:
        print(row)

if __name__ == '__main__':
    main()
