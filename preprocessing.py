from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import os
import csv
import pandas as pd
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
class Preprocessing:
    def __init__(self):
        self.tokenizer = self.create_tokenizer()

        self.tokenized_ds = None
        self.dataset = None
        self.data_collator = None
        pass

    def prepare_i_connect_db(self, train_num=26, test_num=3):

        csv_file_path = './data/craft-hobbies/subjects_basic_info.csv'
        samples_path = './data/craft-hobbies/'
        test_pos_num = test_num // 2
        test_neg_num = test_num // 2
        train_set_array = []
        test_set_array = []
        trainn_set_array = []
        testt_set_array = []
        df_trans = pd.read_csv('./data/craft-hobbies/craft_fold0.csv')
        for k in range(10):

            test_sub_y = df_trans[df_trans.fold == k].iloc[:, 0].tolist()
            train_sub_y = df_trans[df_trans.fold != k].iloc[:, 0].tolist()
            tr_num_subject = 0
            with open(csv_file_path, mode='r') as file:
                sub_test = ""
                sub_train = ""
                csvFile = csv.reader(file)
                for i, lines in enumerate(csvFile):
                    flag=0
                    if i == 0: continue  # header
                    subject_name = lines[0]
                    if not os.path.exists(os.path.join(samples_path, subject_name)):
                        continue
                    label = int(lines[5])
                    tr_file = os.path.join(samples_path, subject_name) + '/' + subject_name + 'p.txt'
                    if subject_name in test_sub_y:
                        sub_test=subject_name
                        flag=1
                    if subject_name in train_sub_y:
                        sub_train=subject_name
                    if flag==1:
                        row_array = self._create_row_index(subject_name=sub_test, label=label,
                                                           tr_file=tr_file)
                        test_set_array += row_array
                    else:
                        row_array = self._create_row_index(subject_name=sub_train, label=label,
                                                           tr_file=tr_file)
                        train_set_array += row_array
                        tr_num_subject += 1
                # size_tr=len(trainn_set_array)
                # size_ts=len(testt_set_array)
                # df_trans = pd.read_csv('transformer_ST_acc_seq46_sh11_st6.csv')
                # for i in range(10):
                #     test_sub_y = df_trans[df_trans.fold == i].iloc[:, 0].tolist()
                #     train_sub_y = df_trans[df_trans.fold != i].iloc[:, 0].tolist()
                #     k=0
                #     l=0
                #     for j in range((size_tr)):
                #         if trainn_set_array[j]['subject_id'] in train_sub_y:
                #
                #             train_set_array=trainn_set_array[j]
                #             print(train_set_array)
                #             k=k+1
                #     for j in range((size_ts)):
                #         if testt_set_array[j]['subject_id'] in test_sub_y:
                #             test_set_array[l]=testt_set_array[j]
                #             l=l+1
                #             print(train_set_array[l]['subject_id'])
                # #print(test_set_array[2])

                pass
        # finalize dataset:
        ds = {}
        ds['train'] = train_set_array
        ds['test'] = test_set_array
        #
        d = {'train': Dataset.from_dict({'label': [train_set_array[i]['label'] for i in range(len(train_set_array))],
                                         'text': [train_set_array[i]['text'] for i in range(len(train_set_array))],
                                         'subject_id': [train_set_array[i]['subject_id'] for i in
                                                        range(len(train_set_array))],
                                         }),
             # 'val': Dataset.from_dict({'label': y_val, 'text': x_val}),
             'test': Dataset.from_dict({'label': [test_set_array[i]['label'] for i in range(len(test_set_array))],
                                        'text': [test_set_array[i]['text'] for i in range(len(test_set_array))],
                                        'subject_id': [test_set_array[i]['subject_id'] for i in
                                                       range(len(test_set_array))],
                                        })
             }
        self.dataset = DatasetDict(d)
        # tokenize
        self.tokenized_ds = self.dataset.map(self._preprocess_function, batched=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="tf")
        print('DATASET Created !')

    def _create_row_index(self, subject_name, label, tr_file):
        lista = pd.read_csv(tr_file, sep=",", header=None)[0].tolist()
        subject_arr = []
        for item in lista:
            item = item.replace('\'', ' ')
            item = item.replace('"', ' ')
            row_item = {}
            row_item['text'] = item
            row_item['label'] = label
            row_item['subject_id'] = subject_name
            subject_arr.append(row_item)
        return subject_arr

    def test_imdb_ds(self):
        imdb = load_dataset("imdb")
        tokenized_imdb = imdb.map(self._preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="tf")
        pass

    def create_tokenizer(self):
        return AutoTokenizer.from_pretrained("roberta-base")

    def _preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)