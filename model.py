from functools import partial as part
from torch import nn
from torch.nn import functional as F
import encs


class CudaCheckableMixin(object):
    @property
    def is_cuda(self):
        # Simple hack to see if the module is built with CUDA parameters.
        if hasattr(self, '__cuda_flag_cache'):
            return self.__cuda_flag_cache
        self.__cuda_flag_cache = next(self.parameters()).is_cuda
        return self.__cuda_flag_cache


class Memory(CudaCheckableMixin, nn.Module):
    def __init__(self,
                 vocabulary_size,
                 embedding_size,
                 sentence_size,
                 memory_size,
                 embedding=None,
                 temporal_embedding=None,):
        super().__init__()
        # Memory configurations.
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.sentence_size = sentence_size
        self.memory_size = memory_size

        # Memory Embeddings.
        self.embedding = (
            embedding or
            nn.Embedding(vocabulary_size, embedding_size)
        )
        self.temporal_embedding = (
            temporal_embedding or
            nn.Embedding(memory_size, embedding_size)
        )

    def _embed(self, x):
        return self\
            .embedding(x.view(-1, self.sentence_size))\
            .view(
                -1,
                self.memory_size,
                self.sentence_size,
                self.embedding_size
            )

    def forward(self, x):
        position_encoding = encs.position_encoding(
            self.embedding_size,
            self.sentence_size,
            cuda=self.is_cuda,
        )
        temporal_encoding = encs.temporal_encoding(
            self.memory_size,
            self.temporal_embedding,
            cuda=self.is_cuda,
        )
        return (position_encoding * self._embed(x)).sum(2) + temporal_encoding


class MemN2N(CudaCheckableMixin, nn.Module):
    WEIGHT_TYING_SCHEMES = ADJACENT, LAYER_WISE, _ = (
        'adjacent', 'layerwise', None
    )

    def __init__(self,
                 vocabulary_hash,
                 vocabulary_size,
                 embedding_size,
                 sentence_size, memory_size, hops=3,
                 weight_tying_scheme=ADJACENT):
        # Model Configurations.
        super().__init__()
        self.vocabulary_hash = vocabulary_hash
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.sentence_size = sentence_size
        self.memory_size = memory_size
        self.hops = hops
        self.weight_tying_scheme = weight_tying_scheme

        # Validate Configurations.
        assert self.weight_tying_scheme in self.WEIGHT_TYING_SCHEMES, (
            'Available weight tying schemes are: {schemes}'
            .format(schemes=self.WEIGHT_TYING_SCHEMES)
        )

        # Memories.
        self.A_memories = nn.ModuleList()
        self.C_memories = nn.ModuleList()
        for i in range(hops):
            # Check if there's any previous memory layer.
            y = i > 0
            A_prev = self.A_memories[i-1] if y else None
            C_prev = self.C_memories[i-1] if y else None

            # 2.2.1. Adjacent Weight Tying
            if self.tied_adjacent:
                A_embedding = C_prev.embedding if y else None
                C_embedding = None
                A_temporal_embedding = C_prev.temporal_embedding if y else None
                C_temporal_embedding = None
            # 2.2.2. Layer-Wise Weight Tying
            elif self.tied_layers:
                A_embedding = A_prev.embedding if y else None
                C_embedding = C_prev.embedding if y else None
                A_temporal_embedding = A_prev.temporal_embedding if y else None
                C_temporal_embedding = C_prev.temporal_embedding if y else None
            # No Weight Tying
            else:
                A_embedding = None
                C_embedding = None
                A_temporal_embedding = None
                C_temporal_embedding = None

            self.A_memories.append(Memory(
                self.vocabulary_size,
                self.embedding_size,
                self.sentence_size,
                self.memory_size,
                embedding=A_embedding,
                temporal_embedding=A_temporal_embedding,
            ))
            self.C_memories.append(Memory(
                self.vocabulary_size,
                self.embedding_size,
                self.sentence_size,
                self.memory_size,
                embedding=C_embedding,
                temporal_embedding=C_temporal_embedding
            ))

        # Affine layer & Query embedding layer.
        if self.tied_adjacent:
            deepest_embedding = self.C_memories[-1].embedding
            first_embedding = self.A_memories[0].embedding
            self.linear = part(F.linear, weight=deepest_embedding.weight)
            self.query_embedding = first_embedding
        else:
            self.linear = nn.Linear(self.embedding_size, self.vocabulary_size)
            self.query_embedding = nn.Embedding(
                self.vocabulary_size,
                self.embedding_size
            )

    def forward(self, x, q, return_encoded=False):
        # Encode the very first query.
        u = self._embed_query(q)
        # Reason through the memories.
        for A, C in zip(self.A_memories, self.C_memories):
            m = A(x)
            c = C(x)
            p = self._match_between_query_and_input_memory(u, m)
            o = self._response_from_output_memory(p, c)
            u = o + u
        # Return the encoded features or estimated scores of the vocabulary.
        return o + u if return_encoded else self.linear(o + u)

    def _embed_query(self, q):
        position_encoding = encs.position_encoding(
            self.embedding_size,
            self.sentence_size,
            cuda=self.is_cuda,
        )
        return (position_encoding * self.query_embedding(q)).sum(1)

    def _match_between_query_and_input_memory(self, u, m):
        return F.softmax((m * u.unsqueeze(1).expand_as(m)).sum(2))

    def _response_from_output_memory(self, p, c):
        return (p.unsqueeze(2).expand_as(c) * c).sum(1)

    @property
    def name(self):
        return (
            'MemN2N'
            '-{hops}hops{weight_tying_scheme}'
            '-{vocabulary_size}vocab-{sentence_size}len-{memory_size}mem'
            '-{embedding_size}emb-{vocabulary_hash}'
        ).format(
            hops=self.hops,
            weight_tying_scheme=(
                '-'+self.weight_tying_scheme if self.weight_tying_scheme else
                ''
            ),
            vocabulary_size=self.vocabulary_size,
            sentence_size=self.sentence_size,
            memory_size=self.memory_size,
            embedding_size=self.embedding_size,
            vocabulary_hash=self.vocabulary_hash,
        )

    @property
    def tied_adjacent(self):
        return self.weight_tying_scheme == self.ADJACENT

    @property
    def tied_layers(self):
        return self.weight_tying_scheme == self.LAYER_WISE
