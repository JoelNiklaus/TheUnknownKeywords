import os
import random
from pathlib import Path

from labels import id2label, label2id, idx_to_labels_list

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # do this to remove gpu with full memory (MUST be before torch import)
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # used for disabling warning (BUT: if deadlock occurs, remove this)

from transformers import EvaluationStrategy, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

from datasets import load_dataset
from train_transformer import train_transformer_pipeline

from util import make_reproducible, compute_metrics, get_prediction_ids
import fire
import wandb
import torch
from pprint import pprint

import pandas as pd

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"

train_transformer_pipeline(DATA_DIR)  # preprocess csv files

extension = ".json"

train_path = DATA_DIR / ("train_trans" + extension)
validation_path = DATA_DIR / ("validation_trans" + extension)
test_path = DATA_DIR / ("test_trans" + extension)


def run(base_model="bert-base-german-cased", fine_tuned_checkpoint_name=None,
        dataset="joelito/sem_eval_2010_task_8",
        input_col_name="MailComplete", label_col_name="ServiceProcessed",
        num_train_epochs=50, do_train=False, do_eval=False, do_predict=True, test_set_sub_size=None, seed=42, ):
    """
    Runs the specified transformer model
    :param base_model:             the name of the base model from huggingface transformers (e.g. roberta-base)
    :param fine_tuned_checkpoint_name:  the name of the fine tuned checkpoint (e.g. checkpoint-500)
    :param dataset:                the name of the dataset from huggingface datasets (e.g. joelito/sem_eval_2010_task_8)
    :param num_train_epochs:            number of epochs to train for
    :param do_train:                    whether to train the model
    :param do_eval:                     whether to evaluate the model in the end
    :param do_predict:                  whether to do predictions on the test set in the end
    :param test_set_sub_size:           make faster by only selecting small subset, otherwise just set to False/None
    :param seed:                        random seed for reproducibility
    :return:
    """
    wandb.init()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    local_model_name = f"{dir_path}/{base_model}-local"

    make_reproducible(seed)
    training_args = TrainingArguments(
        output_dir=f'{local_model_name}/results',  # output directory
        num_train_epochs=num_train_epochs,  # total number of training epochs
        # max_steps=10,  # Set to a small positive number to test models (training is short)
        per_device_train_batch_size=6,  # batch size per device during training
        per_device_eval_batch_size=6,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=f'{local_model_name}/logs',  # directory for storing logs
        logging_steps=10,
        save_steps=500,
        eval_steps=100,
        evaluation_strategy=EvaluationStrategy.STEPS,
        seed=seed,
        run_name=base_model,  # used for wandb
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    print("Loading Dataset")
    # data = load_dataset('csv', data_files={'train': [train_path], 'validation': [validation_path], 'test': [test_path]}, delimiter=";")
    #data = load_dataset('json', data_files={'train': [train_path], 'validation': [validation_path], 'test': [test_path]},field="data")
    data = load_dataset('json', data_files={'train': [train_path], 'validation': [train_path], 'test': [test_path]}, field="data")

    model_path = base_model
    if fine_tuned_checkpoint_name:
        model_path = f"{training_args.output_dir}/{fine_tuned_checkpoint_name}"

    print("Loading Model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, id2label=id2label, label2id=label2id,
                                                               finetuning_task=dataset)
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Tokenizing Dataset")
    # supervised_keys = data['train'].supervised_keys  # 'sentence' and 'relation'
    data = data.map(lambda ex: tokenizer(ex[input_col_name], truncation=True, padding='max_length'),
                    batched=True)
    data.rename_column_(original_column_name=label_col_name,
                        new_column_name='label')  # IMPORTANT: otherwise the loss cannot be computed
    data.set_format(type='pt', columns=['input_ids', 'attention_mask', 'label'], output_all_columns=True)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=data['train'],  # training dataset
        eval_dataset=data['validation'],  # evaluation dataset
        compute_metrics=compute_metrics,  # additional metrics to the loss
    )

    if do_train:
        print("Training on train set")
        trainer.train()

        trainer.save_model(training_args.output_dir)
        # For convenience, we also re-save the tokenizer to the same directory,
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if do_eval:
        print("Evaluating on validation set")
        metrics = trainer.evaluate()
        print(metrics)

    if do_predict:
        print(f"Predicting on test set")

        data['test'].remove_columns_(['label'])

        if test_set_sub_size:
            # IMPORTANT: This command somehow may delete some features in the dataset!
            data['test'] = data['test'].select(indices=range(test_set_sub_size))

        # save inputs because they will be removed by trainer.predict()
        ids = data['test'][0:]['Id']
        subjects = data['test'][0:]['MailSubject']
        textBody = data['test'][0:]['MailTextBody']

        predictions, label_ids, metrics = trainer.predict(data['test'])
        # rename metrics entries to test_{} for wandb
        test_metrics = {}
        for old_key in metrics:
            new_key = old_key.replace("eval_", "test/")
            test_metrics[new_key] = metrics[old_key]
        print(test_metrics)
        wandb.log(test_metrics)

        prediction_ids = get_prediction_ids(predictions)  # get ids of predictions
        predicted_labels = [idx_to_labels_list[prediction_id] for prediction_id in
                            prediction_ids]  # get labels of predictions
        # correct_labels = [idx_to_labels_list[label_id] for label_id in label_ids]  # get labels of ground truth

        # create submissions csv file
        df = pd.DataFrame(list(zip(ids, predicted_labels)),
                          columns=['Id', 'Predicted'])
        df.to_csv("submission2.csv", index=False)

        examples = random.sample(range(data['test'].num_rows), 5)  # look at five random examples from the dataset
        for i in examples:
            print(f"\nId: {ids[i]}")
            print(f"Subject: {subjects[i]}")
            print(f"Text: {textBody[i]}")
            print(f"Predicted Label: {predicted_labels[i]}")
            # print(f"Ground Truth Relation: {correct_labels[i]}")


if __name__ == '__main__':
    fire.Fire(run)
