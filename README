# MemN2N PyTorch Implementation

PyTorch implementation of [End-To-End Memory Networks, NIPS 2015](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)

![model-architecture](./arts/architecture.png)


## Installation
```
$ git clone https://github.com:kuc2477/dl-papers && cd dl-papers
$ pip install -r requirements.txt
```


## CLI
Implementation CLI is provided by `main.py`.

#### Usage 
```
$ ./main.py --help
$ usage: End-to-End Memory Network PyTorch Implementation [-h]
                                                        [--vocabulary-size VOCABULARY_SIZE]
                                                        [--embedding-size EMBEDDING_SIZE]
                                                        [--sentence-size SENTENCE_SIZE]
                                                        [--memory-size MEMORY_SIZE]
                                                        [--hops HOPS]
                                                        [--weight-tying-scheme {adjacent,layerwise,None}]
                                                        [--babi-dataset-name BABI_DATASET_NAME]
                                                        [--babi-tasks BABI_TASKS [BABI_TASKS ...]]
                                                        [--epochs EPOCHS]
                                                        [--test-size TEST_SIZE]
                                                        [--batch-size BATCH_SIZE]
                                                        [--weight-decay WEIGHT_DECAY]
                                                        [--grad-clip-norm GRAD_CLIP_NORM]
                                                        [--lr LR]
                                                        [--lr-decay LR_DECAY]
                                                        [--lr-decay-epochs LR_DECAY_EPOCHS [LR_DECAY_EPOCHS ...]]
                                                        [--checkpoint-interval CHECKPOINT_INTERVAL]
                                                        [--eval-log-interval EVAL_LOG_INTERVAL]
                                                        [--loss-log-interval LOSS_LOG_INTERVAL]
                                                        [--gradient-log-interval GRADIENT_LOG_INTERVAL]
                                                        [--model-dir MODEL_DIR]
                                                        [--dataset-dir DATASET_DIR]
                                                        [--resume-best | --resume-latest]
                                                        [--best] [--no-gpus]
                                                        (--train | --test)

```

#### Train
```
$ python -m visom.server &
$ ./main.py --train [--resume-latest | --resume-best]
```

#### Test
```
$ ./main.py --test
```


## Reference
- [End-To-End Memory Networks, NIPS 2015](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)


## Author
Ha Junsoo / [@kuc2477](https://github.com/kuc2477) / MIT License
