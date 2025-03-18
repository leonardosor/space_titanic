from train_model import train_model


def predictions(model, test_data_path):

    final_model, best_params, study = train_model(args.train_data_path, args.n_trials)
    