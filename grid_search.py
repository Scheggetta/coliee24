from embedding_head import *
from utils import set_random_seeds
import json
import os
from pathlib import Path
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

# TODO: do gris search on both train() and evaluate_model()
SEEDS = [62, 7, 1984]
EPOCHS = 25
SAVE_BEST_WEIGHTS = True
METRIC = 'precision'


def create_training(training_dataloader, q_dataloader, d_dataloader):
    def black_box_function(lr=LR, pe_weight=PE_WEIGHT, factor=FACTOR, threshold=THRESHOLD, patience=PATIENCE,
                           cooldown=COOLDOWN, hidden_units=HIDDEN_UNITS, cosine_loss_margin=COSINE_LOSS_MARGIN,
                           max_docs=MAX_DOCS, ratio_max_similarity=RATIO_MAX_SIMILARITY, pe_cutoff=PE_CUTOFF):
        model = EmbeddingHead(hidden_units=hidden_units).to('cuda')
        scores = []
        for s in SEEDS:
            set_random_seeds(s)
            scores.append(max(train(model, training_dataloader, (q_dataloader, d_dataloader),
                                    num_epochs=EPOCHS, lr=lr, pe_weight=pe_weight, factor=factor, threshold=threshold,
                                    max_docs=int(max_docs), patience=patience, cooldown=cooldown, metric=METRIC,
                                    cosine_loss_margin=cosine_loss_margin, ratio_max_similarity=ratio_max_similarity,
                                    pe_cutoff=pe_cutoff, save_weights=False, verbose=True)[METRIC]))
            print(f'F1 score for seed {s}: {scores[-1]}')
        return sum(scores) / len(scores)

    return black_box_function


if __name__ == '__main__':
    json_path = Path.joinpath(Path('Dataset'), Path(f'task1_train_labels_2024.json'))
    json_dict = json.load(open(json_path))
    split_ratio = 0.9
    train_dict, val_dict = split_dataset(json_dict, split_ratio=split_ratio)
    print(f'Building dataset with split ratio {split_ratio}...')

    training_embeddings = get_gpt_embeddings(folder_path=Path.joinpath(Path('Dataset'), Path('gpt_embed_train')),
                                             selected_dict=train_dict)
    validation_embeddings = get_gpt_embeddings(folder_path=Path.joinpath(Path('Dataset'), Path('gpt_embed_train')),
                                               selected_dict=val_dict)

    dataset = TrainingDataset(training_embeddings, train_dict)
    training_dataloader = DataLoader(dataset, collate_fn=custom_collate_fn, batch_size=32, shuffle=False)

    query_dataset = QueryDataset(validation_embeddings, val_dict)
    document_dataset = DocumentDataset(validation_embeddings, val_dict)
    q_dataloader = DataLoader(query_dataset, batch_size=1, shuffle=False)
    d_dataloader = DataLoader(document_dataset, batch_size=64, shuffle=False)

    pbounds = {'lr': (0.0001, 0.001),
               'threshold': (0.0001, 0.005),
               'patience': (3, 7),
               'cosine_loss_margin': (0.1, 0.6),
               'ratio_max_similarity': (0.9, 0.99),
               }

    optimizer = BayesianOptimization(
        f=create_training(training_dataloader, q_dataloader, d_dataloader),
        pbounds=pbounds,
        random_state=62,
    )
    print(f'Beginning optimization of {len(pbounds)} params with {len(SEEDS)} seeds...')
    optimizer.maximize(
        init_points=3,
        n_iter=6,
    )

    print("Best parameters combination: \n\t{}".format(optimizer.max))

    plt.figure(figsize=(15, 5))
    plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")
    plt.grid(True)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Black box function", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Grid search results for parameters: " + ", ".join(list(pbounds.keys())), fontsize=16)
    plt.show(block=False)
    plt.pause(0.05)

    plot_dir = Path.joinpath(Path('Checkpoints'), Path('GridPlots'))
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(Path.joinpath(plot_dir, Path(f'Plot_for_GD_Target_{optimizer.max["target"]}.png')))

    if SAVE_BEST_WEIGHTS:
        print('Saving best weights...')
        model = EmbeddingHead().to('cuda')
        train(model, training_dataloader, (q_dataloader, d_dataloader), **optimizer.max['params'],
              num_epochs=EPOCHS * 2, save_weights=True, verbose=True)
