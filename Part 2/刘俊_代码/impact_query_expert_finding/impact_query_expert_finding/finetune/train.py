# encoding: utf-8
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample

import impact_query_expert_finding.data.sets as  sets
import logging
import impact_query_expert_finding.data.io as io
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


def train(config, work_dir, triples_dir, triples_name, version, encoder_name):
    train_batch_size = 64
    num_epochs = 4
    dataset = sets.DataSet("aminer")
    data_path = os.path.join(work_dir, version, "dataset_associations")
    dataset.load(data_path)
    T = dataset.ds.documents

    model_save_path = data_path + '/output/' + encoder_name

    # model_name = 'allenai/scibert_scivocab_uncased'
    # model_name = 'distilbert-base-uncased'
    # model_name = 'roberta-base'
    model_name = 'nfliu/scibert_basevocab_uncased'
    embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[embedding_model, pooling_model])

    logging.info("Read Triplet train dataset")
    train_samples = []
    dev_samples = []

    triples = io.load_as_json(triples_dir, triples_name)
    for i, row in enumerate(triples):
        if i % 20 == 0:
            dev_samples.append(
                InputExample(texts=[str(T[int(row[0])]), str(T[int(row[1])]), str(T[int(row[2])])], label=0))

        train_samples.append(
            InputExample(texts=[str(T[int(row[0])]), str(T[int(row[1])]), str(T[int(row[2])])], label=0))

    train_dataset = SentencesDataset(train_samples, model=model)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    # train_loss = losses.TripletLoss(model=model, triplet_margin=5)
    train_loss = losses.TripletLoss(model=model, triplet_margin=8)

    evaluator = TripletEvaluator.from_input_examples(dev_samples, name='dev')
    warmup_steps = int(len(train_dataset) * num_epochs / train_batch_size * 0.1)  # 10% of train data

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)

    test_evaluator = TripletEvaluator.from_input_examples(dev_samples, name='test')
    model.evaluate(test_evaluator)

    print("model_save_path : ", model_save_path)
