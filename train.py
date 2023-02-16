import os
import evaluate
import numpy as np
from transformers import create_optimizer
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from transformers.keras_callbacks import KerasMetricCallback
from transformers.keras_callbacks import PushToHubCallback
from transformers import pipeline
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
import torch
from transformers import AutoConfig
import torch.nn.functional as F
from transformers import RobertaForMaskedLM

class Train:
    def __init__(self, dataset, tokenized_ds, data_collator, tokenizer):
        self.tokenized_ds = tokenized_ds
        self.dataset = dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        pass

    def _compute_metrics(self, eval_pred):
        accuracy = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return accuracy.compute(predictions=predictions, references=labels)

    def _id2label(self):
        return {0: "MCI", 1: "Healthy"}

    def _label2id(self):
        return {"MCI":0, "Healthy":1}

    def train_model(self):

        batch_size = 29
        num_epochs = 1
        batches_per_epoch = len(self.tokenized_ds["train"]) // batch_size
        total_train_steps = int(batches_per_epoch * num_epochs)
        optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

        '''load model'''
        model = TFAutoModelForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2, id2label=self._id2label(), label2id=self._label2id()
        )
        print("it is runned for tokenizing each sentence")
        #add lines to generate word ebmeddings on roberta to encode each sentence
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        # # Import tokenizer from transformers package
        # from transformers import BertTokenizer
        #
        # # Load the tokenizer of the "bert-base-cased" pretrained model
        # # See https://huggingface.co/transformers/pretrained_models.html for other models
        # tz = BertTokenizer.from_pretrained("bert-base-cased")
        #
        # # The senetence to be encoded
        # sent = "Let's learn deep learning!"
        #
        # # Encode the sentence
        # encoded = tz.encode_plus(
        #     text=sent,  # the sentence to be encoded
        #     add_special_tokens=True,  # Add [CLS] and [SEP]
        #     max_length=64,  # maximum length of a sentence
        #     pad_to_max_length=True,  # Add [PAD]s
        #     return_attention_mask=True,  # Generate the attention mask
        #     return_tensors='pt',  # ask the function to return PyTorch tensors
        # )
        #
        # # Get the input IDs and attention mask in tensor format
        # input_ids = encoded['input_ids']
        # attn_mask = encoded['attention_mask']
        # x=self.dataset['train']
        # print(tokenizer.tokenize(self.dataset['train']['text']))
        #
        # ##the format of a encoding
        # print(tokenizer.batch_encode_plus([self.dataset['train']['text']]))
        #
        # ##op wants the input id
        # print(tokenizer.batch_encode_plus([self.dataset['train']['text']])['input_ids'])
        #
        # ##op wants the input id without first and last token
        # print(tokenizer.batch_encode_plus([self.dataset['train']['text']])['input_ids'][0][1:-1])
        '''create tf record'''
        tf_train_set = model.prepare_tf_dataset(
            self.tokenized_ds["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=self.data_collator,
        )

        tf_validation_set = model.prepare_tf_dataset(
            self.tokenized_ds["test"],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=self.data_collator,
        )
        '''compile model'''
        model.compile(optimizer=optimizer)
        ''''''
        metric_callback = KerasMetricCallback(metric_fn=self._compute_metrics, eval_dataset=tf_validation_set)
        # push_to_hub_callback = PushToHubCallback(
        #     output_dir="./model_save",
        #     tokenizer=self.tokenizer
        # )
        # callbacks = [metric_callback, push_to_hub_callback]
        #
        # checkpoint_filepath = './model_train/checkpoint'
        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     save_weights_only=True,
        #     monitor='val_accuracy',
        #     mode='max',
        #     save_best_only=False)

        callbacks = [metric_callback]

        '''fit'''
        model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=num_epochs, callbacks=callbacks)
        '''save weights'''


        #model.save("./model_save")
        model.save_pretrained("./model_save")
        print("hhh")
        model.save_weights("./model_save/Roberta_weights.h5")

        pass

    def test_model(self):
        text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

        #tokenizer = AutoTokenizer.from_pretrained("./model_save",config=AutoConfig.from_pretrained("./model_save"))

        tokenizer = AutoTokenizer.from_pretrained("./model_save",from_pt=True)
        inputs = tokenizer(text, return_tensors="tf")

        model = TFAutoModelForSequenceClassification.from_pretrained("./model_save",from_pt=True)
        logits = model(**inputs).logits
        predictions=tf.nn.sigmoid(logits)

        print(predictions)
        predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])

        model.config.id2label[predicted_class_id]
        pass

    def eval_model(self):
        # init
        tokenizer = AutoTokenizer.from_pretrained("./model_save",from_pt=True)
        model = TFAutoModelForSequenceClassification.from_pretrained("./model_save",from_pt=True)
        #
        p_all_samples = []
        g_all_samples = []
        #
        s_names, s_labels = self._return_subject_names(self.dataset['test'])
        subject_base_pre = {}
        label_base_pre = {}
        for i in range(len(s_names)):
            subject_base_pre[s_names[i]] = {'subject_id': s_names[i], 'label': s_labels[i], 'g': [], 'p': []}
        # load samples
        for item in self.dataset['test']:
            input = tokenizer(item['text'], return_tensors="tf")
            logits = model(**input).logits
            p_id = int(tf.math.argmax(logits, axis=-1)[0])
            # for i in range(len(s_names)):
            #     print(s_names[i])
            # print("\n")
            # predictions = tf.math.softmax(logits, axis=-1)
            # print(predictions)
            g_id = item['label']
            s_id = item['subject_id']
            g_all_samples.append(g_id)
            p_all_samples.append(p_id)
            # label-based:
            subject_base_pre[s_id]['g'].append(g_id)
            subject_base_pre[s_id]['p'].append(p_id)
        # confusion matrix all:
        cm = confusion_matrix(g_all_samples, p_all_samples, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot()
        plt.savefig('r.png')

        cm_norm = confusion_matrix(g_all_samples, p_all_samples, labels=[0, 1], normalize='true')
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=[0, 1])
        disp_norm.plot()
        plt.savefig('n.png')
        precision, recall, fscore, support = score(g_all_samples, p_all_samples)
        acc  = accuracy_score(g_all_samples, p_all_samples)
        print('acc => ' + str(acc * 100))
        print('precision: {}'.format(precision) + '-' + str(np.mean(precision)*100))
        print('recall: {}'.format(recall) + '-' + str(np.mean(recall)*100))
        print('fscore: {}'.format(fscore) + '-' + str(np.mean(fscore)*100))
        print('support: {}'.format(support))
        #
        for s_id in s_names:
            label = subject_base_pre[s_id]['label']
            p_arr = subject_base_pre[s_id]['p']
            g_arr = subject_base_pre[s_id]['g']
            acc = accuracy_score(g_arr, p_arr)
            print(s_id + ' -> Label: ' + str(label) + ', acc: ' + str(acc) + '%')


    def _return_subject_names(self, t_arr):
        s_names = []
        s_labels = []
        for item in t_arr:
            sn = item['subject_id']
            if sn not in s_names:
                s_names.append(sn)
                s_labels.append(item['label'])
        return s_names, s_labels


# probs = torch.stack(logits.scores, dim=1).softmax(-1)
# prediction=logits.softmax(dim=-1)[0]
# print(probs)


#logits = model.predict()
