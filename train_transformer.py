import argparse
import pickle
import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("train")

from speechfulagent import ExplainerTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", default=".")

    parser.add_argument("--model-dir", default=".")

    args = parser.parse_args()

    trainer = ExplainerTrainer(
        args.dataset_dir,
        37
    )
    tokenizer, history, explainer, train_info = trainer.train()
    logger.info("saving model")
    explainer.save_model(args.model_dir, train=train_info)
    with open("history.pkl", "wb") as f:
        pickle.dump(history, f)
