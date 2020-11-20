import os
import random
from pathlib import Path

import fire
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # do this to remove gpu with full memory (MUST be before torch import)
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # used for disabling warning (BUT: if deadlock occurs, remove this)

from transformers import EvaluationStrategy, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

from datasets import load_dataset

from util import make_reproducible, compute_metrics, get_prediction_ids

from pprint import pprint

label_set = ['EDA_ANW_ARIS (EDA Scout)', 'EDA_ANW_ARS Remedy', 'EDA_ANW_CH@World (MOSS)', 'EDA_ANW_CodX PostOffice',
             'EDA_ANW_DMS Fabasoft eGov Suite', 'EDA_ANW_EDA PWC Tool', 'EDA_ANW_EDAContacts', 'EDA_ANW_EDAssist+',
             'EDA_ANW_FDFA Security App', 'EDA_ANW_IAM Tool EDA', 'EDA_ANW_ITDoc Sharepoint', 'EDA_ANW_Internet EDA',
             'EDA_ANW_Intranet/Collaboration EDA', 'EDA_ANW_MOVE!', 'EDA_ANW_NOS:4', 'EDA_ANW_ORBIS',
             'EDA_ANW_Office Manager', 'EDA_ANW_Plato-HH', 'EDA_ANW_Reisehinweise', 'EDA_ANW_SAP Services',
             'EDA_ANW_SysP eDoc', 'EDA_ANW_ZACWEB', 'EDA_ANW_Zeiterfassung SAP', 'EDA_ANW_Zentrale Raumreservation EDA',
             'EDA_ANW_at Honorarvertretung', 'EDA_ANW_eVERA', 'EDA_S_APS', 'EDA_S_APS_Monitor', 'EDA_S_APS_OS_BasisSW',
             'EDA_S_APS_PC', 'EDA_S_APS_Peripherie', 'EDA_S_Arbeitsplatzdrucker', 'EDA_S_BA_2FA', 'EDA_S_BA_Account',
             'EDA_S_BA_Datenablage', 'EDA_S_BA_Internetzugriff', 'EDA_S_BA_Mailbox', 'EDA_S_BA_RemoteAccess',
             'EDA_S_BA_ServerAusland', 'EDA_S_BA_UCC_Benutzertelefonie', 'EDA_S_BA_UCC_IVR', 'EDA_S_Backup & Restore',
             'EDA_S_Benutzerunterstützung', 'EDA_S_Betrieb Übermitttlungssysteme', 'EDA_S_Büroautomation',
             'EDA_S_IT Sicherheit', 'EDA_S_Mobile Kommunikation', 'EDA_S_Netzdrucker', 'EDA_S_Netzwerk Ausland',
             'EDA_S_Netzwerk Inland', 'EDA_S_Order Management', 'EDA_S_Peripheriegeräte', 'EDA_S_Raumbewirtschaftung',
             'EDA_S_Zusätzliche Software', '_Pending']

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = ROOT_DIR / "data"
train_path = DATASET_DIR / "train_trans.csv"
test_path = DATASET_DIR / "test_trans.csv"


def run(base_model="distilbert-base-german-cased", fine_tuned_checkpoint_name=None,
        dataset="joelito/sem_eval_2010_task_8",
        input_col_name="MailTextBody", label_col_name="ServiceProcessed",
        num_train_epochs=10, do_train=False, do_eval=False, do_predict=True, test_set_sub_size=None, seed=42, ):
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

    dir_path = os.path.dirname(os.path.realpath(__file__))
    local_model_name = f"{dir_path}/{base_model}-local"

    make_reproducible(seed)
    training_args = TrainingArguments(
        output_dir=f'{local_model_name}/results',  # output directory
        num_train_epochs=num_train_epochs,  # total number of training epochs
        max_steps=-1,  # Set to a small positive number to test models (training is short)
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
    data = load_dataset('csv', data_files={'train': [train_path], 'test': [test_path]}, delimiter=";")
    pprint(data)

    idx_to_labels_list = label_set  # list to look up the label indices
    id2label = {k: v for k, v in enumerate(idx_to_labels_list)}
    label2id = {v: k for k, v in enumerate(idx_to_labels_list)}

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
        if test_set_sub_size:
            # IMPORTANT: This command somehow may delete some features in the dataset!
            data['test'] = data['test'].select(indices=range(test_set_sub_size))

        # save sentences because they will be removed by trainer.predict()
        sentences = data['test'][0:-1]['sentence']

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
        correct_labels = [idx_to_labels_list[label_id] for label_id in label_ids]  # get labels of ground truth

        examples = random.sample(range(data['test'].num_rows), 5)  # look at five random examples from the dataset
        for i in examples:
            print(f"\nSentence: {sentences[i]}")
            print(f"Predicted Relation: {predicted_labels[i]}")
            print(f"Ground Truth Relation: {correct_labels[i]}")


if __name__ == '__main__':
    fire.Fire(run)
