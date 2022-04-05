import matchzoo as mz
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import keras

train_dataset = pd.read_csv('../../train_data/train_data_T.csv', delimiter=',')
validation_dataset = pd.read_csv('../../train_data/validation_data_T.csv', delimiter=',')

train_dataset = train_dataset.replace(np.nan, ' ', regex=True)
validation_dataset = validation_dataset.replace(np.nan, ' ', regex=True)

list_data1 = []
for i, row in train_dataset.iterrows():
    
    line1 = {'id_left': str(row['article_id']),
            'text_left':str(row['article_page_title'])+" "+str(row['article_meta_description'])+" "+str(row['article_keywords']),
            'id_right':str(row['table_id']),
            'text_right':str(row['table_page_title'])+" "+str(row['table_page_summary'])+" "+str(row['table_page_keywords']),
            'label':row['label']
           }
    
    list_data1.append(line1)

df1 = pd.DataFrame(list_data1)
train_pack = mz.pack(df1)


list_data2 = []
for i, row in validation_dataset.iterrows():
    
    line2 = {'id_left': str(row['article_id']),
            'text_left':str(row['article_page_title'])+" "+str(row['article_meta_description'])+" "+str(row['article_keywords']),
            'id_right':str(row['table_id']),
            'text_right':str(row['table_page_title'])+" "+str(row['table_page_summary'])+" "+str(row['table_page_keywords']),
            'label':row['label']
           }
    
    list_data2.append(line2)

df2 = pd.DataFrame(list_data2)
valid_pack = mz.pack(df2)

ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss())
ranking_task.metrics = [mz.metrics.Precision()]

glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)

preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=130, fixed_length_right=130, remove_stop_words=True)
train_pack_processed = preprocessor.fit_transform(train_pack, verbose=0)
valid_pack_processed = preprocessor.transform(valid_pack, verbose=0)

model = mz.models.DUET()
model.params.update(preprocessor.context)
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = 300
model.params['lm_filters'] = 32
model.params['lm_hidden_sizes'] = [32]
model.params['dm_filters'] = 32
model.params['dm_kernel_size'] = 3
model.params['dm_d_mpool'] = 4
model.params['dm_hidden_sizes'] = [32]
model.params['dropout_rate'] = 0.5
optimizer = keras.optimizers.Adamax(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.params['optimizer'] = 'adagrad'
model.guess_and_fill_missing_params()
model.build()
model.compile()
model.backend.summary()

embedding_matrix = glove_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
model.load_embedding_matrix(embedding_matrix)

pred_x, pred_y = valid_pack_processed[:].unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=32,model_save_path='DUET_result_title_main_passage_keywords/DUET_title_main_passage_keywords', once_every=1)


train_generator = mz.DataGenerator(
    train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1,
    batch_size=32
)

history = model.fit_generator(train_generator, epochs=100, callbacks=[evaluate], verbose=0)