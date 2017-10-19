import functools
import itertools
import contextlib
import collections
import copy
import hashlib
import re
import operator
import os
import os.path
import shutil
import multiprocessing
from multiprocessing.pool import Pool
from tempfile import NamedTemporaryFile
import requests
from tqdm import tqdm
import numpy as np
import torch
from torch import LongTensor
from torch.utils.data import Dataset


_CPU_COUNT = multiprocessing.cpu_count()


@contextlib.contextmanager
def _progress(m1, m2=None, end1='\n', end2='\n', flush1=True, flush2=False):
    print(m1, end=end1, flush=flush1)
    yield
    if m2 is not None:
        print(m2, end=end2, flush=flush2)


class BabiQA(Dataset):
    _UNKNOWN = '<UNK>'
    _PADDING = '<PAD>'
    _DIRNAME = 'babi'
    _CHROME_UA = (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_3) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/27.0.1453.93 Safari/537.36'
    )
    _URL = (
        'http://www.thespermwhale.com/jaseweston/'
        'babi/tasks_1-20_v1-2.tar.gz'
    )

    def __init__(self,
                 dataset_name='en-10k',
                 tasks=None,
                 vocabulary=None,
                 vocabulary_size=200,
                 sentence_size=20,
                 sentence_number=None,
                 train=True, download=True, fresh=False, path='./datasets'):
        self._dataset_name = dataset_name
        self._tasks = tasks or [i+1 for i in range(20)]
        self._train = train
        self._path = os.path.join(path, self._DIRNAME)
        self._path_to_dataset = os.path.join(self._path, self._dataset_name)
        self._path_to_preprocessed = os.path.join(
            self._path + '-preprocessed', self._dataset_name
        )

        if download:
            self._download()

        self._sentence_size = sentence_size
        self._sentence_number = sentence_number
        self._vocabulary = vocabulary or self._generate_vocabulary(
            self._tasks, vocabulary_size-2, train=self._train
        )
        self._word2idx = {w: i+2 for i, w in enumerate(self._vocabulary)}
        self._word2idx[self._UNKNOWN] = self.unknown_idx
        self._word2idx[self._PADDING] = self.padding_idx
        self._idx2word = {i+2: w for i, w in enumerate(self._vocabulary)}
        self._idx2word[self.unknown_idx] = self._UNKNOWN
        self._idx2word[self.padding_idx] = self._PADDING

        self._paths = self._load_to_disk(
            self._tasks,
            sentence_size=self._sentence_size,
            sentence_number=self._sentence_number,
            train=self._train, fresh=fresh,
        )

    def __getitem__(self, index):
        path = self._paths[index]
        loaded = np.load(path)
        return (
            torch.from_numpy(loaded['sentences']),
            torch.from_numpy(loaded['query']),
            torch.from_numpy(loaded['answer']),
        )

    def __len__(self):
        return len(self._paths)

    @classmethod
    def collate_fn(cls, samples):
        x, q, a = zip(*samples)
        return (
            LongTensor(torch.stack(x)),
            LongTensor(torch.stack(q)),
            LongTensor(torch.stack(a)).squeeze(1),
        )

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def vocabulary_hash(self):
        key = '/'.join([*self._vocabulary])
        return hashlib.sha256(key.encode()).hexdigest()[:10]

    @property
    def dataset_hash(self):
        key = '/'.join([*self._vocabulary])
        key += '-{}'.format(self._sentence_size)
        key += '-{}'.format(self._sentence_number)
        return hashlib.sha256(key.encode()).hexdigest()[:10]

    @property
    def vocabulary_size(self):
        return len(self._vocabulary) + 2

    @property
    def sentence_size(self):
        return self._sentence_size

    @property
    def unknown_idx(self):
        return 0

    @property
    def padding_idx(self):
        return 1

    def word2idx(self, word):
        return self._word2idx.get(word, self.unknown_idx)

    def idx2word(self, idx):
        return self._idx2word[idx]

    @staticmethod
    def _remove_ending_punctuation(w):
        return w.split('?')[0].split('.')[0]

    def _cleanup_disk(self):
        for path in self._paths:
            os.unlink(path)

    def _load_to_disk(self, tasks, sentence_size, sentence_number=None,
                      train=True, fresh=False, pool_size=_CPU_COUNT*3):
        try:
            with _progress('=> Preprocessing the data... ', ''):
                pool = Pool(pool_size)
                partial = functools.partial(
                    self._load_task_data_to_disk,
                    sentence_size=sentence_size,
                    sentence_number=sentence_number,
                    train=train, fresh=fresh,
                )
                return list(itertools.chain(*pool.map(partial, tasks)))
        except (KeyboardInterrupt, SystemExit, SystemError, Exception):
            # TODO: NOT IMPLEMENTED YET
            raise

    def _load_task_data_to_disk(self, task,
                                sentence_size,
                                sentence_number=None, train=True, fresh=False):
        dirpath = os.path.join(
            self._path_to_preprocessed,
            '{task}-{train}-{dataset_hash}'.format(
                task=task, train=('train' if train else 'test'),
                dataset_hash=self.dataset_hash
            )
        )

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        elif not fresh and os.listdir(dirpath):
            print('  * Using preprocessed data in {}'.format(dirpath))
            return [os.path.join(dirpath, n) for n in os.listdir(dirpath)]

        paths = []
        data = self._task_data(task, train=train)
        parsed = self._parse(data)
        for i, x in enumerate(parsed):
            # Load the data to the disk and retain their paths instaed of
            # loading them to the memory directly. This is to prevent OOM
            # error. The temporary files will be discarded on exit by the
            # `~_cleanup_disk()` callback.
            tmp = NamedTemporaryFile(delete=False, dir=dirpath)
            sentences, query, answer = self._encode_x(
                x,
                sentence_size=sentence_size,
                sentence_number=sentence_number,
            )
            np.savez(
                tmp,
                sentences=sentences,
                query=query,
                answer=answer,
            )
            tmp.close()
            paths.append(tmp.name)
        return paths

    def _encode_x(self, x, sentence_size, sentence_number=None):
        assert sentence_number is None or sentence_number > 0, (
            'Sentence number has to be a positive integer or None'
        )

        sentences, query, answer = x

        # Calculate difference between number of given sentences and target
        # sentence number. This is to pad or truncate the sentences to fit the
        # required number of sentences.
        diff = (
            sentence_number-len(sentences)
            if sentence_number is not None else 0
        )
        sentences_to_pad = max(0, diff)
        sentences_to_truncate = max(0, -diff)

        # Pad or truncate the sentences to fit the target sentence number.
        if sentences_to_pad > 0:
            null_sentence = ' '.join([self._UNKNOWN] * sentence_size)
            null_sentences = [null_sentence] * sentences_to_pad
            sentences = sentences + null_sentences
        elif sentences_to_truncate > 0:
            sentences = sentences[sentences_to_truncate:]

        encoded_sentences = np.array([
            self._encode_words(*s.split(), sentence_size=sentence_size) for
            s in sentences
        ])
        encoded_query = self._encode_words(
            query, sentence_size=sentence_size
        )
        encoded_answer = self._encode_words(
            answer, sentence_size=sentence_size
        )[0, None]

        return encoded_sentences, encoded_query, encoded_answer

    def _encode_words(self, *words, sentence_size, sentence_number=None):
        paddings = (self._PADDING,) * (sentence_size-len(words))
        indices = np.array([
            self.word2idx(self._remove_ending_punctuation(w)) for
            w in words + paddings
        ])
        return indices.astype(np.int64)

    def _generate_vocabulary(self, tasks, vocabulary_size, train=True,
                             pool_size=_CPU_COUNT*3):
        with _progress('=> Generating a vocabulary... '):
            counters = Pool(pool_size).map(functools.partial(
                self._generate_task_vocabulary,
                vocabulary_size=vocabulary_size, train=train
            ), tasks)
        counter = functools.reduce(operator.add, counters)
        vocabulary = [w for w, c in counter.most_common(vocabulary_size)]
        vocabulary.sort()
        return tuple(vocabulary)

    def _generate_task_vocabulary(self, task, vocabulary_size, train=True):
        words = self._task_data(task, train=train).split()
        words = [
            self._remove_ending_punctuation(w) for w
            in words if not w.isdigit()
        ]
        return collections.Counter(words)

    def _parse(self, data):
        sentences = []
        still_in_the_same_story = False

        for l in data.splitlines():
            i, l = re.split(' ', l, maxsplit=1)
            i, l = int(i), l.strip()
            still_in_the_same_story = int(i) != 1
            try:
                query, answer_and_supports = l.strip().split('?')
                query = query.strip()
                answer, *supports = answer_and_supports.strip().split()
                yield copy.copy(sentences), query, answer
            except ValueError:
                if not still_in_the_same_story:
                    sentences.clear()
                sentences.append(l)

    def _task_data(self, task, train=True):
        startswith = 'qa{task}_'.format(task=task)
        endswith = '_{train_or_test}.txt'.format(train_or_test=(
            'train' if train else 'test'
        ))

        try:
            with open([
                    os.path.join(self._path_to_dataset, n) for n in
                    os.listdir(self._path_to_dataset) if
                    n.startswith(startswith) and
                    n.endswith(endswith)
            ][0]) as f:
                return f.read()

        except IndexError:
            raise FileNotFoundError

    def _download(self):
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        # Check if the dataset already exists or not.
        elif os.listdir(self._path):
            print('=> Using the dataset in "{dir}" for "{target}"'.format(
                dir=self._path, target='{dataset_name}-{train}'
                .format(
                    dataset_name=self._dataset_name,
                    train=('train' if self._train else 'test')
                )
            ))
            return

        stream = requests.get(
            self._URL, stream=True, timeout=3,
            headers={'user-agent': self._CHROME_UA}
        )

        if not stream.ok:
            raise RuntimeError(
                '=> {url} returned {code} for following reason: {reason}'
                .format(
                    url=self._URL,
                    code=stream.status_code,
                    reason=stream.reason
                ))

        total_size = int(stream.headers.get('content-length', 0))
        chunk_size = 1024 * 32
        stream_with_progress = tqdm(
            stream.iter_content(chunk_size=chunk_size),
            total=total_size//chunk_size,
            unit='KiB',
            unit_scale=True,
            desc='=> Downloading a bAbI QA dataset',
        )

        download_path = self._path + '.tar.gz'
        with open(download_path, 'wb') as f:
            for chunk in stream_with_progress:
                f.write(chunk)
            shutil.copyfileobj(stream.raw, f)
            stream.close()

        with _progress('=> Extracting... '):
            os.system('tar -xzf {tar} --strip-components=1 -C {dest}'.format(
                tar=download_path, dest=self._path
            ))
            os.remove(download_path)
