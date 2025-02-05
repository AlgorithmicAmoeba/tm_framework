import string
from typing import List, Union

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map  # or thread_map
from tqdm import tqdm
from pathlib import Path
# from octis.dataset.dataset import Dataset
from collections import Counter


class Dataset:
    """
    Dataset handles a dataset and offers methods to access, save and edit the dataset data
    """

    def __init__(self, corpus=None, vocabulary=None, labels=None, metadata=None, document_indexes=None):
        """
        Initialize a dataset, parameters are optional
        if you want to load a dataset, initialize this
        class with default values and use the load method
        Parameters
        ----------
        corpus : corpus of the dataset
        vocabulary : vocabulary of the dataset
        labels : labels of the dataset
        metadata : metadata of the dataset
        """
        self.__corpus = corpus
        self.__vocabulary = vocabulary
        self.__metadata = metadata
        self.__labels = labels
        self.__original_indexes = document_indexes
        self.dataset_path = None
        self.is_cached = False

    def get_corpus(self):
        return self.__corpus

    # Partitioned Corpus getter
    def get_partitioned_corpus(self, use_validation=True):
        if "last-training-doc" in self.__metadata:
            last_training_doc = self.__metadata["last-training-doc"]
            if use_validation:
                last_validation_doc = self.__metadata["last-validation-doc"]
                if self.__corpus is not None and last_training_doc != 0:
                    train_corpus = []
                    test_corpus = []
                    validation_corpus = []

                    for i in range(last_training_doc):
                        train_corpus.append(self.__corpus[i])
                    for i in range(last_training_doc, last_validation_doc):
                        validation_corpus.append(self.__corpus[i])
                    for i in range(last_validation_doc, len(self.__corpus)):
                        test_corpus.append(self.__corpus[i])
                    return train_corpus, validation_corpus, test_corpus
            else:
                if self.__corpus is not None and last_training_doc != 0:
                    if "last-validation-doc" in self.__metadata.keys():
                        last_validation_doc = self.__metadata["last-validation-doc"]
                    else:
                        last_validation_doc = 0

                    train_corpus = []
                    test_corpus = []
                    for i in range(last_training_doc):
                        train_corpus.append(self.__corpus[i])

                    if last_validation_doc != 0:
                        for i in range(last_validation_doc, len(self.__corpus)):
                            test_corpus.append(self.__corpus[i])
                    else:
                        for i in range(last_training_doc, len(self.__corpus)):
                            test_corpus.append(self.__corpus[i])
                    return train_corpus, test_corpus
        else:
            return [self.__corpus]


    # Edges getter
    def get_edges(self):
        return self.__edges

    # Labels getter
    def get_labels(self):
        return self.__labels

    # Metadata getter
    def get_metadata(self):
        return self.__metadata

    # Info getter
    def get_info(self):
        if "info" in self.__metadata:
            return self.__metadata["info"]
        else:
            return None

    # Vocabulary getter
    def get_vocabulary(self):
        return self.__vocabulary

    def _save_metadata(self, file_name):
        """
        Saves metadata in json serialized format
        Parameters
        ----------
        file_name : name of the file to write
        Returns
        -------
        True if the data is saved
        """
        data = self.get_metadata()
        if data is not None:
            with open(file_name, 'w') as outfile:
                json.dump(data, outfile)
                return True
        else:
            raise Exception("error in saving metadata")

    def _load_metadata(self, file_name):
        """
        Loads metadata from json serialized format
        Parameters
        ----------
        file_name : name of the file to read
        """
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as metadata_file:
                metadata = json.load(metadata_file)
            self.__metadata = metadata

    def _load_corpus(self, file_name):
        """
        Loads corpus from a file
        Parameters
        ----------
        file_name : name of the file to read
        """
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as corpus_file:
                corpus = [line.strip().split() for line in corpus_file]
            self.__corpus = corpus
        else:
            raise Exception("error in loading corpus")

    def _save_edges(self, file_name):
        """
        Saves edges in a file, a line for each document
        Parameters
        ----------
        file_name : name of the file to write
        """
        data = self.get_edges()
        if data is not None:
            with open(file_name, 'w') as outfile:
                for element in data:
                    outfile.write("%s\n" % element)
        else:
            raise Exception("error in saving edges")

    def _load_edges(self, file_name):
        """
        Loads edges from a file
        Parameters
        ----------
        file_name : name of the file to read
        """
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as edges_file:
                edges = [line[0:len(line) - 1] for line in edges_file]
            self.__edges = edges

    def _save_labels(self, file_name):
        """
        Saves the labels in a file, each line contains
        the labels of a single document
        Parameters
        ----------
        file_name : name of the file to write
        """
        data = self.get_labels()
        if data is not None:
            with open(file_name, 'w') as outfile:
                for element in data:
                    outfile.write("%s\n" % json.dumps(element))
        else:
            raise Exception("error in saving labels")

    def _load_labels(self, file_name):
        """
        Loads labels from a file
        Parameters
        ----------
        file_name : name of the file to read
        ----------
        """
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as labels_file:
                labels = [json.loads(line.strip()) for line in labels_file]
            self.__labels = labels

    def _save_vocabulary(self, file_name):
        """
        Saves vocabulary dictionary in a file
        Parameters
        ----------
        file_name : name of the file to write
        -------
        """
        data = self.get_vocabulary()
        if data is not None:
            with open(file_name, 'w', encoding='utf8') as outfile:
                for word in data:
                    outfile.write(word + "\n")
        else:
            raise Exception("error in saving vocabulary")

    def _save_document_indexes(self, file_name):
        """
        Saves document indexes in a file
        Parameters
        ----------
        file_name : name of the file to write
        -------
        """
        if self.__original_indexes is not None:
            with open(file_name, 'w') as outfile:
                for i in self.__original_indexes:
                    outfile.write(str(i) + "\n")

    def _load_vocabulary(self, file_name):
        """
        Loads vocabulary from a file
        Parameters
        ----------
        file_name : name of the file to read
        """
        vocabulary = []
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as vocabulary_file:
                for line in vocabulary_file:
                    vocabulary.append(line.strip())
            self.__vocabulary = vocabulary
        else:
            raise Exception("error in loading vocabulary")

    def _load_document_indexes(self, file_name):
        """
        Loads document indexes from a file
        Parameters
        ----------
        file_name : name of the file to read
        """
        document_indexes = []
        file = Path(file_name)
        if file.is_file():
            with open(file_name, 'r') as indexes_file:
                for line in indexes_file:
                    document_indexes.append(line.strip())
            self.__original_indexes = document_indexes
        else:
            raise Exception("error in loading vocabulary")

    def save(self, path, multilabel=False):
        """
        Saves all the dataset info in a folder
        Parameters
        ----------
        path : path to the folder in which files are saved.
               If the folder doesn't exist it will be created
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        try:
            partitions = self.get_partitioned_corpus()
            corpus, partition = [], []
            for i, p in enumerate(partitions):
                if i == 0:
                    part = 'train'
                elif i == 1 and len(partitions) == 3:
                    part = 'val'
                else:
                    part = 'test'

                for doc in p:
                    corpus.append(' '.join(doc))
                    partition.append(part)

            df = pd.DataFrame(data=corpus)
            df = pd.concat([df, pd.DataFrame(partition)], axis=1)

            if multilabel:
                labs = [' '.join(lab) for lab in self.__labels]
            else:
                labs = self.__labels
            if self.__labels:
                df = pd.concat([df, pd.DataFrame(labs)], axis=1)
            df.to_csv(path + '/corpus.tsv', sep='\t', index=False, header=False)

            self._save_vocabulary(path + "/vocabulary.txt")
            self._save_metadata(path + "/metadata.json")
            self._save_document_indexes(path + "/indexes.txt")
            self.dataset_path = path

        except:
            raise Exception("error in saving the dataset")

    def load_custom_dataset_from_folder(self, path, multilabel=False):
        """
        Loads all the dataset from a folder
        Parameters
        ----------
        path : path of the folder to read
        """
        self.dataset_path = path
        try:
            if exists(self.dataset_path + "/metadata.json"):
                self._load_metadata(self.dataset_path + "/metadata.json")
            else:
                self.__metadata = dict()
            df = pd.read_csv(
                self.dataset_path + "/corpus.tsv", sep='\t', header=None)
            if len(df.keys()) > 1:
                # just make sure docs are sorted in the right way (train - val - test)
                final_df = pd.concat(
                    [df[df[1] == 'train'],
                     df[df[1] == 'val'],
                     df[df[1] == 'test']])
                self.__metadata['last-training-doc'] = len(
                    final_df[final_df[1] == 'train'])
                self.__metadata['last-validation-doc'] = len(
                    final_df[final_df[1] == 'val']) + len(
                        final_df[final_df[1] == 'train'])

                self.__corpus = [d.split() for d in final_df[0].tolist()]
                if len(final_df.keys()) > 2:
                    if multilabel:
                        self.__labels = [
                            doc.split() for doc in final_df[2].tolist()]
                    else:
                        self.__labels = final_df[2].tolist()

            else:
                self.__corpus = [d.split() for d in df[0].tolist()]
                self.__metadata['last-training-doc'] = len(df[0])

            if exists(self.dataset_path + "/vocabulary.txt"):
                self._load_vocabulary(self.dataset_path + "/vocabulary.txt")
            else:
                vocab = set()
                for d in self.__corpus:
                    for w in set(d):
                        vocab.add(w)
                self.__vocabulary = list(vocab)
            if exists(self.dataset_path + "/indexes.txt"):
                self._load_document_indexes(self.dataset_path + "/indexes.txt")
        except:
            raise Exception("error in loading the dataset:" + self.dataset_path)

    def fetch_dataset(self, dataset_name, data_home=None, download_if_missing=True):
        """Load the filenames and data from a dataset.
        Parameters
        ----------
        dataset_name: name of the dataset to download or retrieve
        data_home : optional, default: None
            Specify a download and cache folder for the datasets. If None,
            all data is stored in '~/octis' subfolders.
        download_if_missing : optional, True by default
            If False, raise an IOError if the data is not locally available
            instead of trying to download the data from the source site.
        """

        data_home = get_data_home(data_home=data_home)
        cache_path = _pkl_filepath(data_home, dataset_name + ".pkz")
        dataset_home = join(data_home, dataset_name)
        cache = None
        if exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    compressed_content = f.read()
                uncompressed_content = codecs.decode(
                    compressed_content, 'zlib_codec')
                cache = pickle.loads(uncompressed_content)
            except Exception as e:
                print(80 * '_')
                print('Cache loading failed')
                print(80 * '_')
                print(e)

        if cache is None:
            if download_if_missing:
                cache = download_dataset(
                    dataset_name, target_dir=dataset_home,
                    cache_path=cache_path)
            else:
                raise IOError(dataset_name + ' dataset not found')
        self.is_cached = True
        self.__corpus = [d.split() for d in cache["corpus"]]
        self.__vocabulary = cache["vocabulary"]
        self.__metadata = cache["metadata"]
        self.dataset_path = cache_path
        self.__labels = cache["labels"]

"""
Maps the language to its corresponding spacy model
"""
spacy_model_mapping = {
    'chinese': 'zh_core_web_sm', 'danish': 'nl_core_news_sm',
    'dutch': 'nl_core_news_sm', 'english': 'en_core_web_sm',
    'french': 'fr_core_news_sm', 'german': 'de_core_news_sm',
    'greek': 'el_core_news_sm', 'italian': 'it_core_news_sm',
    'japanese': 'ja_core_news_sm', 'lithuanian': 'lt_core_news_sm',
    'norwegian': 'nb_core_news_sm', 'polish': 'pl_core_news_sm',
    'portuguese': 'pt_core_news_sm', 'romanian': 'ro_core_news_sm',
    'russian': 'ru_core_news_sm', 'spanish': 'es_core_news_sm'}


class Preprocessing:
    def __init__(
        self, lowercase: bool = True, vocabulary: List[str] = None,
        max_features: int = None, min_df: float = 0.0, max_df: float = 1.0,
        remove_punctuation: bool = True, punctuation: str = string.punctuation,
        remove_numbers: bool = True, lemmatize: bool = True,
        stopword_list: Union[str, List[str]] = None, min_chars: int = 1,
        min_words_docs: int = 0, language: str = 'english', split: bool = True,
        verbose: bool = False, num_processes: int = None,
        save_original_indexes=True, remove_stopwords_spacy: bool = True):
        """
        init Preprocessing

        :param lowercase: if true, words in documents are reduced to
            lowercase (default: true)
        :type lowercase: boolean
        :param vocabulary: the vocabulary of the corpus to preprocess
            (default: None)
        :type vocabulary: list
        :param max_features: maximum number of words that the vocabulary must
            contain. The less frequent words will be removed. If it's not None,
            then max_df and min_df are ignored (default: None)
        :type max_features: int
        :param min_df: words below this minumum document frequency will be
            removed (default: 0.0)
        :type min_df: float
        :param max_df: words above this maximum document frequency will be
            removed (default: 1.0)
        :type max_df: float
        :param remove_punctuation: if true, punctuation will be removed
            (default: true)
        :type remove_punctuation: bool
        :param punctuation: string containing all the punctuation chars that
            need to be removed (default:
        string.punctuation)
        :type punctuation: str
        :param remove_numbers: if true, numbers will be removed
        :type remove_numbers: bool
        :param remove_stopwords_spacy: bool , if true use spacy to remove
            stopwords (default: true)
        :param lemmatize: if true, words will be lemmatized using a spacy model
            according to the language that has been set (default: true)
        :type lemmatize: bool
        :param stopword_list: if a list of strings is passed, the strings will
            be removed from the texts. Otherwise, if a str is passed, it
            represents the language of the stopwords that need to be removed.
            The stopwords are spacy's stopwords (default: None)
        :type stopword_list: str or list of str
        :param min_chars: mininum number of characters that a token should have
            (default: 1)
        :type min_chars: int
        :param min_words_docs: minimun number of words that a document should
            contain (default: 0)
        :type min_words_docs: int
        :param language: language of the documents. It needs to be set for the
            lemmatizer (default: english)
        :type language: str
        :param split: if true, the corpus will be split in train (85%),
            testing (7.5%) and validation (7.5%) set (default: true)
        :type split: bool
        :param verbose: if true, some steps of the preprocessing will be
            printed on screen (default: false)
        :type verbose: bool
        :param num_processes: number of processes to run the preprocessing
        :type num_processes: int
        :param save_original_indexes: if true, it keeps track of the original
            indexes of the documents
        """
        self.vocabulary = vocabulary
        self.lowercase = lowercase
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.remove_punctuation = remove_punctuation
        self.punctuation = punctuation
        self.lemmatize = lemmatize
        self.language = language
        self.num_processes = num_processes
        self.remove_numbers = remove_numbers
        self.save_original_indexes = save_original_indexes

        if self.lemmatize:
            lang = spacy_model_mapping[self.language]
            try:
                self.spacy_model = spacy.load(lang)
            except IOError:
                raise IOError("Can't find model " + lang + ". Check the data directory or download it using the "
                                                           "following command:\npython -m spacy download " + lang)
        self.split = split
        self.verbose = verbose

        self.remove_stopwords_spacy = remove_stopwords_spacy

        stopwords = []
        # if stopwords is None then stopwords are not removed
        if stopword_list is None:
            self.remove_stopwords_spacy = False
        else:
            # if custom list is specified, then we do not use spacy stopwords
            if type(stopword_list) == list:
                stopwords = set(stopword_list)
                self.remove_stopwords_spacy = False
            elif self.remove_stopwords_spacy:
                assert stopword_list == language
            else:
                # if remove_stopwords_spacy is false, then use MALLET English stopwords
                if 'english' in stopword_list:
                    stop_word_path = Path(__file__).parent.joinpath('stopwords', 'english.txt')
                    with open(stop_word_path) as fr:
                        stopwords = [line.strip() for line in fr.readlines()]
                        assert stopword_list == language

        self.stopwords = stopwords
        self.min_chars = min_chars
        self.min_doc_words = min_words_docs
        self.preprocessing_steps = []

    def preprocess_dataset(self, documents_path, labels_path=None, multilabel=False):
        """
        preprocess the input dataset

        :param documents_path: path to the documents file. Each row of the file represents a document
        :type documents_path: str
        :param labels_path: path to the documents file. Each row of the file represents a label. Its index corresponds
        to the index of the documents file (default: None)
        :type labels_path: str
        :param multilabel: if true, a document is supposed to have more than one label (labels are split by whitespace)
        :type multilabel: bool

        :return octis.dataset.dataset.Dataset
        """
        docs = [line.strip() for line in open(documents_path, 'r').readlines()]
        if self.num_processes is not None:
            # with Pool(self.num_processes) as p:
            #    docs = p.map(self.simple_preprocessing_steps, docs)
            chunksize = max(1, len(docs) // (self.num_processes * 20))
            docs_list = process_map(self.simple_preprocessing_steps, docs, max_workers=self.num_processes, chunksize=chunksize)
        else:
            docs = list(map(self.simple_preprocessing_steps, tqdm(docs)))
        if self.lowercase:
            self.preprocessing_steps.append("lowercase")
        if self.remove_punctuation:
            self.preprocessing_steps.append('remove_punctuation')
        if self.lemmatize:
            self.preprocessing_steps.append('lemmatize')

        vocabulary = self.filter_words(docs)
        print("created vocab")
        print(len(vocabulary))
        final_docs, final_labels, document_indexes = [], [], []
        if labels_path is not None:
            if multilabel:
                labels = [
                    line.strip().split()
                    for line in open(labels_path, 'r').readlines()]
            else:
                labels = [
                    line.strip()
                    for line in open(labels_path, 'r').readlines()]

            vocab = set(vocabulary)
            for i, doc, label in zip(range(len(docs)), docs, labels):
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    final_labels.append(label)
                    document_indexes.append(i)

            labels_to_remove = set([k for k, v in dict(
                Counter(final_labels)).items() if v <= 3])
            if len(labels_to_remove) > 0:
                docs = final_docs
                labels = final_labels
                document_indexes, final_labels, final_docs = [], [], []
                for i, doc, label in zip(range(len(docs)), docs, labels):
                    if label not in labels_to_remove:
                        final_docs.append(doc)
                        final_labels.append(label)
                        document_indexes.append(i)
        else:
            vocab = set(vocabulary)
            for i, doc in enumerate(docs):
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    document_indexes.append(i)

        self.preprocessing_steps.append('filter documents with less than ' + str(self.min_doc_words) + " words")
        if self.verbose:
            print("words filtering done")
        metadata = {"total_documents": len(docs), "vocabulary_length": len(vocabulary),
                    "preprocessing-info": self.preprocessing_steps
                    # ,"labels": list(set(final_labels)), "total_labels": len(set(final_labels))
                    }
        if self.split:
            if len(final_labels) > 0:
                train, test, y_train, y_test = train_test_split(
                    range(len(final_docs)), final_labels, test_size=0.15, random_state=1, shuffle=True)#stratify=final_labels)

                train, validation = train_test_split(train, test_size=3 / 17, random_state=1, shuffle=True)# stratify=y_train)

                partitioned_labels = [final_labels[doc] for doc in train + validation + test]
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]
                document_indexes = [document_indexes[doc] for doc in train + validation + test]
                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)
                if self.save_original_indexes:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata,
                                   labels=partitioned_labels, document_indexes=document_indexes)
                else:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata,
                                   labels=partitioned_labels)
            else:
                train, test = train_test_split(range(len(final_docs)), test_size=0.15, random_state=1)
                train, validation = train_test_split(train, test_size=3 / 17, random_state=1)

                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]
                document_indexes = [document_indexes[doc] for doc in train + validation + test]
                if self.save_original_indexes:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                                   document_indexes=document_indexes)
                else:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                                   document_indexes=document_indexes)
        else:
            if self.save_original_indexes:
                return Dataset(final_docs, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                               document_indexes=document_indexes)
            else:

                return Dataset(final_docs, vocabulary=vocabulary, metadata=metadata, labels=final_labels)

    def filter_words(self, docs):
        if self.vocabulary is not None:
            self.preprocessing_steps.append('filter words by vocabulary')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = TfidfVectorizer(df_max_freq=self.max_df, df_min_freq=self.min_df, vocabulary=self.vocabulary,
                                         token_pattern=r"(?u)\b\w{" + str(self.min_chars) + ",}\b",
                                         lowercase=self.lowercase, stop_words=self.stopwords)

        elif self.max_features is not None:
            self.preprocessing_steps.append('filter vocabulary to ' + str(self.max_features) + ' terms')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            # we ignore df_max_freq e df_min_freq because self.max_features is not None
            vectorizer = TfidfVectorizer(lowercase=self.lowercase, max_features=self.max_features,
                                         stop_words=self.stopwords,
                                         token_pattern=r"(?u)\b[\w|\-]{" + str(self.min_chars) + r",}\b")

        else:

            #string.punctuation

            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, lowercase=self.lowercase,
                                         token_pattern=r"(?u)\b[\w|\-]{" + str(self.min_chars) + r",}\b",
                                         stop_words=self.stopwords)

        vectorizer.fit_transform(docs)
        vocabulary = vectorizer.get_feature_names_out()
        return vocabulary

    '''
    def _foo(self, docs, vocabulary, labels_path):
        final_docs, final_labels = [], []
        if labels_path is not None:
            labels = [line.strip() for line in open(labels_path, 'r').readlines()]
            for doc, label in zip(docs, labels):
                new_doc = [w for w in doc.split() if w in set(vocabulary)]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    final_labels.append(label)
            return final_docs, final_labels
        else:
            for doc in docs:
                new_doc = [w for w in doc.split() if w in set(vocabulary)]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
            return final_docs, []
    '''

    def simple_preprocessing_steps(self, doc):
        new_d = doc
        new_d = new_d.replace('\n', '')
        new_d = new_d.replace('\t', '')
        if self.lowercase:
            new_d = new_d.lower()
        if self.lemmatize:
            if self.remove_stopwords_spacy:
                tokens = self.spacy_model(new_d, disable=['tok2vec'])
                filtered_tokens = [token.lemma_ for token in tokens if not token.is_stop]
                new_d = ' '.join(filtered_tokens)
            elif self.stopwords:
                new_d = ' '.join(
                    [token.lemma_ for token in self.spacy_model(new_d) if token.lemma_ not in set(self.stopwords)])
            else:
                new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d)])

        if self.remove_punctuation:
            new_d = new_d.translate(str.maketrans(self.punctuation, ' ' * len(self.punctuation)))
        if self.remove_numbers:
            new_d = new_d.translate(str.maketrans("0123456789", ' ' * len("0123456789")))
        new_d = " ".join(new_d.split())
        return new_d