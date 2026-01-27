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
        pathfile=args.dataset_dir,
        d_state=37,
        d_hidden=128,
        nhead=4,
        dim_feedforward=512,
        num_decoder_layers=2,
        max_iter=350,
        info_every_epoch=10,
        batch_first=True
    )
    tokenizer, history, explainer, train_info = trainer.train()
    logger.info("saving model")
    explainer.save_model(args.model_dir, train=train_info)
    with open("history.pkl", "wb") as f:
        pickle.dump(history, f)
