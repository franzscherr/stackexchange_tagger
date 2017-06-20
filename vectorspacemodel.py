#!/usr/bin/python
# __________________________________________________________________________________________________
# Process StackExchange data dump into vector space model
#

import numpy as np
import pickle
import scipy.sparse as sp
from lxml import etree
import os

from loader import Loader


class StackExchangeVectorSpace(object):
    def __init__(self, directory_path='', post_limit=10000000,
                 significance_limit_min=5, significance_limit_max_factor=0.95):
        """
        Convert StackExchange data dump into vector space model
        
        :param directory_path: Root directory that contains the data dump. Converted data will be stored herein.
        :param post_limit: Maximum number of posts to convert
        :param significance_limit_min: Minimum number of occurrences to consider a term relevant
        :param significance_limit_max_factor: Maximum fraction that enables a term to be relevant
        """
        self.posts_term_matrix = None
        self.tags_matrix = None
        self.nonzero_tags = None
        self.inverse_vocabulary = None
        self.inverse_tag_vocabulary = None

        if not os.path.exists(os.path.join(directory_path, 'vocabulary_and_frequency.pkl')):
            # iterative loading
            self.inverse_vocabulary = {}
            self.inverse_vocabulary_length = 0
            self.document_frequency = []
            self.inverse_tag_vocabulary = {}
            self.inverse_tag_vocabulary_length = 0
            self.tag_frequency = []

            # document term matrix sparse builder
            self.document_term_indices = []
            self.document_term_index_pointer = [0]
            self.document_term_data = []

            # tag matrix sparse builder
            self.tag_indices = []
            self.tag_index_pointer = [0]
            self.tag_data = []

            context = etree.iterparse(os.path.join(directory_path, 'Posts.xml'), tag='row')
            loader = Loader(limit=post_limit)
            loader.load(context, callback=self.posts_process_callback)
            del context
            del loader
            self.term_document_matrix = sp.csr_matrix((self.document_term_data,
                                                       self.document_term_indices,
                                                       self.document_term_index_pointer))
            del self.document_term_data
            del self.document_term_indices
            del self.document_term_index_pointer
            self.target_tag_matrix = sp.csr_matrix((self.tag_data,
                                                    self.tag_indices,
                                                    self.tag_index_pointer))
            del self.tag_data
            del self.tag_indices
            del self.tag_index_pointer

            self.tag_frequency = np.array(self.tag_frequency)
            self.document_frequency = np.array(self.document_frequency)
            inverse_document_frequency = \
                np.log(self.term_document_matrix.shape[0] / self.document_frequency).reshape((1, -1))

            self.model_matrix = self.term_document_matrix.multiply(sp.csr_matrix(inverse_document_frequency))

            sp.save_npz(os.path.join(directory_path, 'model_matrix.npz'), self.model_matrix)
            sp.save_npz(os.path.join(directory_path, 'target_tags.npz'), self.target_tag_matrix)
            sp.save_npz(os.path.join(directory_path, 'term_document_matrix.npz'), self.term_document_matrix)
            with open(os.path.join(directory_path, 'vocabulary_and_frequency.pkl'), 'wb') as f:
                pickle.dump([self.inverse_vocabulary, self.inverse_tag_vocabulary, self.document_frequency, self.tag_frequency], f)
        else:
            self.model_matrix = sp.load_npz(os.path.join(directory_path, 'model_matrix.npz'))
            self.target_tag_matrix = sp.load_npz(os.path.join(directory_path, 'target_tags.npz'))
            self.term_document_matrix = sp.load_npz(os.path.join(directory_path, 'term_document_matrix.npz'))
            with open(os.path.join(directory_path, 'vocabulary_and_frequency.pkl'), 'rb') as f:
                self.inverse_vocabulary, self.inverse_tag_vocabulary, self.document_frequency, self.tag_frequency = pickle.load(f)

        n_documents = self.term_document_matrix.shape[0]
        min_significance = significance_limit_min
        max_significance = int(significance_limit_max_factor * n_documents)
        # consider only those terms that have valid occurrence count
        valid_column_indices = np.where(np.logical_and(self.document_frequency > min_significance,
                                                       self.document_frequency < max_significance))[0]
        # build vocabularies
        self.vocabulary = {}
        self.tag_vocabulary = {}
        self.build_vocabularies(valid_column_indices)

        self.term_document_matrix = self.term_document_matrix[:, valid_column_indices]
        self.non_zero_tags = np.array(self.target_tag_matrix.sum(axis=1) > 3, dtype=np.bool).reshape((-1,))
        self.data = self.model_matrix[:, valid_column_indices]
        self.supervised_data = self.data[self.non_zero_tags, :]
        self.unsupervised_data = self.data[np.logical_not(self.non_zero_tags), :]
        # self.supervised_data = self.model_matrix[self.non_zero_tags, :]
        # self.supervised_data = self.supervised_data[:, valid_column_indices]
        self.target = self.target_tag_matrix[self.non_zero_tags, :]

    def build_vocabularies(self, valid_column_indices):
        """
        Update inverse vocabularies based on the valid column indices, build vocabularies
        
        :param valid_column_indices: Column indices (terms) that have valid occurrence count/are valid
        """
        word_list = [None] * len(self.inverse_vocabulary)
        for word, index in self.inverse_vocabulary.items():
            word_list[index] = word
        self.inverse_vocabulary = {}
        for new_index, column in enumerate(valid_column_indices):
            self.vocabulary[new_index] = word_list[column]
            self.inverse_vocabulary[word_list[column]] = new_index
        for tag, index in self.inverse_tag_vocabulary.items():
            self.tag_vocabulary[index] = tag

    def posts_process_callback(self, posts_list):
        """
        Callback for the loader to process preprocessed posts on the fly
        
        :param posts_list: Converted posts to append to vector space model
        """
        for post in posts_list:
            words = post[0] + post[1]
            word_set = set(words)
            tags = post[2]

            # update inverse_tag_vocabulary / increase tag_frequency / update tag matrix
            for tag in tags:
                if tag not in self.inverse_tag_vocabulary:
                    self.inverse_tag_vocabulary[tag] = self.inverse_tag_vocabulary_length
                    self.inverse_tag_vocabulary_length += 1
                    self.tag_frequency.append(1)
                else:
                    self.tag_frequency[self.inverse_tag_vocabulary[tag]] += 1

                # append to tag matrix builder
                self.tag_indices.append(self.inverse_tag_vocabulary[tag])
                self.tag_data.append(1)

            # update vocabulary / increase document_frequency / update document term matrix
            for word in words:
                if word not in self.inverse_vocabulary:
                    self.inverse_vocabulary[word] = self.inverse_vocabulary_length
                    self.inverse_vocabulary_length += 1
                    self.document_frequency.append(1)
                elif word in word_set:
                    self.document_frequency[self.inverse_vocabulary[word]] += 1
                    word_set.remove(word)

                # append to document term matrix builder
                self.document_term_indices.append(self.inverse_vocabulary[word])
                self.document_term_data.append(1)
            self.document_term_index_pointer.append(len(self.document_term_indices))
            self.tag_index_pointer.append(len(self.tag_indices))
            del post
        del posts_list


def main():
    import argparse
    parser = argparse.ArgumentParser(description='process a StackExchange data dump into vectorspace model')
    parser.add_argument('--directory', type=str, help='root directory of StackExchange data dump', required=True)
    parser.add_argument('--limit', type=int, help='maximum number of posts to be processed')
    args = parser.parse_args()

    if args.limit is None:
        vsm = StackExchangeVectorSpace(directory_path=args.directory)
    else:
        vsm = StackExchangeVectorSpace(directory_path=args.directory, post_limit=args.limit)

    print('-----------------------')
    print('-- Data Set Summary ---')
    print('Tagged posts:    {:6d}'.format(vsm.supervised_data.shape[0]))
    print('Total posts:     {:6d}'.format(vsm.model_matrix.shape[0]))
    print('--')
    print('Vocabulary size: {:6d}'.format(vsm.model_matrix.shape[1]))
    print('Number of tags:  {:6d}'.format(vsm.target.shape[1]))
    print('-----------------------')


if __name__ == '__main__':
    main()
