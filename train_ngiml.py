from tools.train_ngiml import parse_args, run_training


if __name__ == "__main__":
    configuration = parse_args()
    run_training(configuration)
