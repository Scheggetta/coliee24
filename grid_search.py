import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

from embedding_head import *
from utils import set_random_seeds


SEEDS = [2, 1984]
EPOCHS = 30
RETRAIN_ON_BEST = False
METRIC = 'val_f1_score'

OPTIMIZER_SEED = 42
INIT_POINTS = 1
N_ITER = 0

PBOUNDS = {}

if METRIC in ['precision', 'recall', 'val_f1_score']:
    lr_scheduler_mode = 'max'
elif METRIC in ['train_loss', 'val_loss']:
    lr_scheduler_mode = 'min'
else:
    raise ValueError('Invalid metric')


def create_training(training_dataloader, qd_dataloader):
    def black_box_function(hidden_units=HIDDEN_UNITS,
                           emb_out=EMB_OUT,
                           dropout_rate=DROPOUT_RATE,
                           lr=LR,
                           pe_weight=PE_WEIGHT,
                           factor=FACTOR,
                           threshold=THRESHOLD,
                           patience=PATIENCE,
                           cooldown=COOLDOWN,
                           cosine_loss_margin=COSINE_LOSS_MARGIN,
                           dynamic_cutoff=DYNAMIC_CUTOFF,
                           max_docs=MAX_DOCS,
                           ratio_max_similarity=RATIO_MAX_SIMILARITY,
                           pe_cutoff=PE_CUTOFF,
                           sample_size=SAMPLE_SIZE):

        scores = []
        for s in SEEDS:
            set_random_seeds(s)

            model = (EmbeddingHead(hidden_units=int(hidden_units), emb_out=int(emb_out), dropout_rate=dropout_rate)
                     .to('cuda'))
            _optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            _lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer, mode=lr_scheduler_mode, factor=factor,
                                                                       patience=int(patience), cooldown=int(cooldown),
                                                                       threshold=threshold)

            set_sample_size(int(sample_size))
            scores.append(max(train(model=model,
                                    train_dataloader=training_dataloader,
                                    validation_dataloader=qd_dataloader,
                                    num_epochs=EPOCHS,
                                    optimizer=_optimizer,
                                    # loss_function=_loss_function,          # DEFAULT
                                    cosine_loss_margin=cosine_loss_margin,
                                    lr_scheduler=_lr_scheduler,
                                    lr_scheduler_mode=lr_scheduler_mode,
                                    # val_loss_function=_val_loss_function,  # DEFAULT
                                    # score_function=_score_function,        # DEFAULT
                                    dynamic_cutoff=dynamic_cutoff,
                                    pe_weight=pe_weight,
                                    max_docs=int(max_docs),
                                    ratio_max_similarity=ratio_max_similarity,
                                    pe_cutoff=int(pe_cutoff),
                                    metric=METRIC,
                                    save_weights=True,
                                    verbose=True
                                    )[METRIC]
                              )
                          )
            print(f'Target metric score for seed {s}: {scores[-1]:.6f}')
        return sum(scores) / len(scores)

    return black_box_function


if __name__ == '__main__':
    training_dataloader, qd_dataloader = create_dataloaders('train')

    pbounds = PBOUNDS

    optimizer = BayesianOptimization(
        f=create_training(training_dataloader, qd_dataloader),
        pbounds=pbounds,
        random_state=OPTIMIZER_SEED,
    )
    print(f'Beginning optimization of {len(pbounds)} params with {len(SEEDS)} seeds...')
    optimizer.maximize(
        init_points=INIT_POINTS,
        n_iter=N_ITER,
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

    # Convert to int the parameters that should be int of the best combination
    should_be_int = ['hidden_units', 'emb_out', 'sample_size', 'patience', 'cooldown', 'pe_cutoff', 'max_docs']
    best_params = {}
    for param in optimizer.max['params']:
        if param in should_be_int:
            best_params[param] = int(optimizer.max['params'][param])
        else:
            best_params[param] = optimizer.max['params'][param]

    if RETRAIN_ON_BEST:
        print('Re-training on best parameters...')
        model = (EmbeddingHead(hidden_units=best_params['hidden_units'] if 'hidden_units' in best_params else HIDDEN_UNITS,
                               emb_out=best_params['emb_out'] if 'emb_out' in best_params else EMB_OUT,
                               dropout_rate=best_params['dropout_rate'] if 'dropout_rate' in best_params else DROPOUT_RATE)
                 .to('cuda'))

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'] if 'lr' in best_params else LR)
        lr_scheduler = (torch.optim.lr_scheduler.
                        ReduceLROnPlateau(optimizer, mode=lr_scheduler_mode,
                                          factor=best_params['factor'] if 'factor' in best_params else FACTOR,
                                          patience=best_params['patience'] if 'patience' in best_params else PATIENCE,
                                          cooldown=best_params['cooldown'] if 'cooldown' in best_params else COOLDOWN,
                                          threshold=best_params['threshold'] if 'threshold' in best_params else THRESHOLD)
                        )

        train(model,
              training_dataloader,
              qd_dataloader,
              optimizer=optimizer,
              lr_scheduler=lr_scheduler,
              num_epochs=EPOCHS * 2,
              metric=METRIC,
              save_weights=True,
              verbose=True,
              **best_params
              )
