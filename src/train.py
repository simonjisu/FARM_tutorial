import farm
import torch
import argparse
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.data_silo import DataSilo
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.utils import MLFlowLogger

def main(args):
    print(f"[INFO] PyTorch Version: {torch.__version__}")
    print(f"[INFO] FARM Version: {farm.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Devices available: {}".format(device))
    ml_logger = MLFlowLogger(tracking_uri=args.tracking_uri)
    ml_logger.init_experiment(experiment_name=args.experiment_name, run_name=args.run_name)
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        do_lower_case=False
    )
    # TextClassificationProcessor
    processor = TextClassificationProcessor(
        tokenizer=tokenizer,
        train_filename=args.train_filename,
        test_filename=args.test_filename,
        header=0,
        max_seq_len=args.max_seq_len,
        data_dir=args.data_dir,
        label_list=args.label_list,
        metric=args.metric,
        label_column_name=args.label_column_name,
        text_column_name=args.text_column_name
    )
    # DataSilo
    data_silo = DataSilo(
        processor=processor,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    # LanguageModel: Build pretrained language model
    language_model = LanguageModel.load(
        args.pretrained_model_name_or_path, 
        language="korean"
    )
    # TextClassificationHead: Build predictor layer
    prediction_head = TextClassificationHead(
        num_labels=len(args.label_list), 
        class_weights=data_silo.calculate_class_weights(
            task_name="text_classification"
        )
    )
    # AdaptiveModel: Combine all
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=args.embeds_dropout_prob,
        lm_output_types=["per_sequence"],
        device=device
    )
    # Initialize Optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=args.n_epochs
    )
    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        evaluate_every=args.evaluate_every,
        checkpoints_to_keep=args.checkpoints_to_keep,
        checkpoint_root_dir=args.checkpoint_root_dir,
        epochs=args.n_epochs,
        n_gpu=args.n_gpu,
        lr_schedule=lr_schedule,
        device=device, 
    )
    # now train!
    model = trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run farm")
    parser.add_argument("--tracking_uri", type=str, default="https://public-mlflow.deepset.ai/",
        help="MLFlow - tracking uri ")
    parser.add_argument("--experiment_name", type=str, default="FARM_tutorial",
        help="MLFlow - experiment name")
    parser.add_argument("--run_name", type=str, default="NSMC",
        help="MLFlow - run name")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="beomi/kcbert-base",
        help="Tokenizer, LanguageModel - pretrained model name")

    parser.add_argument("--train_filename", type=str, default="train.tsv",
        help="Processor - train file name")
    parser.add_argument("--test_filename", type=str, default="test.tsv",
        help="Processor - test file name")
    parser.add_argument("--max_seq_len", type=int, default=150,
        help="Processor - max sequence lenght of tokens")
    parser.add_argument("--data_dir",  type=str, default="./nsmc/",
        help="Processor - data directory")
    parser.add_argument("--label_list", nargs"*", default=["bad", "good"],
        help="Processor - label list with string")
    parser.add_argument("--metric",  type=str, default="acc",
        help="Processor - acc or f1_macro")
    parser.add_argument("--label_column_name",  type=str, default="label",
        help="Processor - label column name")
    parser.add_argument("--text_column_name",  type=str, default="document",
        help="Processor - text column name")

    parser.add_argument("--batch_size", type=int, default=32,
        help="DataSilo - train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32,
        help="DataSilo - eval batch size")

    parser.add_argument("--embeds_dropout_prob ", type=float, default=0.1,
        help="AdaptiveModel - The probability that a value in the embeddings \
        returned by the language model will be zeroed.")

    parser.add_argument("--learning_rate ", type=float, default=2e-5,
        help="initialize_optimizer - learning rate")
    parser.add_argument("--n_epochs", type=int, default=3,
        help="initialize_optimizer - number of epochs")

    parser.add_argument("--n_gpu", type=int, default=4,
        help="Trainer - number of gpus") 
    parser.add_argument("--checkpoint_root_dir", type=str, default="./ckpt/",
        help="Trainer - checkpoint root directory")
    parser.add_argument("--checkpoints_to_keep", type=int, default=3,
        help="Trainer - number of checkpoint to keep")
    parser.add_argument("--evaluate_every", type=int, default=200,
        help="Trainer - evaluate steps")
    args = parser.parse_args()
    main(args)