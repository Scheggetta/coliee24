from embedding_head import *
import json
import os
from pathlib import Path
from bayes_opt import BayesianOptimization


# TODO: do gris search on both train() and evaluate_model()
PRINT_ALL = False


def create_training(model, training_dataloader, q_dataloader, d_dataloader):
    def black_box_function(lr=LR, pe_weight=PE_WEIGHT, factor=FACTOR, threshold=THRESHOLD, patience=PATIENCE,
                           cooldown=COOLDOWN, num_epochs=2, cosine_loss_margin=COSINE_LOSS_MARGIN,
                           max_docs=MAX_DOCS, ratio_max_similarity=RATIO_MAX_SIMILARITY, pe_cutoff=PE_CUTOFF):
        return max(train(model, training_dataloader, (q_dataloader, d_dataloader), num_epochs,
                         lr=lr, pe_weight=pe_weight, factor=factor, threshold=threshold, max_docs=max_docs,
                         patience=patience, cooldown=cooldown, cosine_loss_margin=cosine_loss_margin,
                         ratio_max_similarity=ratio_max_similarity, pe_cutoff=pe_cutoff,
                         save_weights=False)['val_f1_score'])
    return black_box_function


if __name__ == '__main__':
    json_dict = json.load(open('Dataset/task1_train_labels_2024.json'))
    train_dict, val_dict = split_dataset(json_dict, split_ratio=0.9)

    training_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_train',
                                             selected_dict=train_dict)
    validation_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_train',
                                               selected_dict=val_dict)

    dataset = TrainingDataset(training_embeddings, train_dict)
    training_dataloader = DataLoader(dataset, collate_fn=custom_collate_fn, batch_size=32, shuffle=False)

    query_dataset = QueryDataset(validation_embeddings, val_dict)
    document_dataset = DocumentDataset(validation_embeddings, val_dict)
    q_dataloader = DataLoader(query_dataset, batch_size=1, shuffle=False)
    d_dataloader = DataLoader(document_dataset, batch_size=64, shuffle=False)

    model = EmbeddingHead().to('cuda')

    # Bounded region of parameter space
    pbounds = {'lr': (0.0001, 0.01),
               # 'pe_weight': (0, 1),
               'factor': (0.1, 0.5),
               'threshold': (1e-4, 1e-2),
               'patience': (1, 5),
               'cooldown': (1, 5)}

    optimizer = BayesianOptimization(
        f=create_training(model, training_dataloader, q_dataloader, d_dataloader),
        pbounds=pbounds,
        random_state=62,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )

    if PRINT_ALL:
        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \n\t{}".format(i, res))
    print("Best parameters combination: \n\t{}".format(optimizer.max))
