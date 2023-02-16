from preprocessing import Preprocessing
from train import Train

def preprocess_data():
    pp_obj = Preprocessing()
    pp_obj.prepare_i_connect_db()
    return pp_obj.dataset, pp_obj.tokenized_ds, pp_obj.data_collator, pp_obj.tokenizer
    # pp_obj.test_imdb_ds()
if __name__ == '__main__':
    dataset, tokenized_ds, data_collator, tokenizer = preprocess_data()
    tr_obj = Train(dataset, tokenized_ds, data_collator, tokenizer)

    tr_obj.train_model()
    tr_obj.test_model()
    tr_obj.eval_model()

    pass