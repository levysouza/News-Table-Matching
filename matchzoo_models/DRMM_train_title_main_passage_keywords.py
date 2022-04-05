import matchzoo as mz
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

bin_size = 30
model = mz.models.DRMM()
model.params.update(preprocessor.context)
model.params['input_shapes'] = [[130,], [130, bin_size,]]
model.params['task'] = ranking_task
model.params['mask_value'] = 0
model.params['embedding_output_dim'] = glove_embedding.output_dim
model.params['mlp_num_layers'] = 1
model.params['mlp_num_units'] = 10
model.params['mlp_num_fan_out'] = 1
model.params['mlp_activation_func'] = 'tanh'
model.params['optimizer'] = 'adadelta'
model.build()
model.compile()
model.backend.summary()

embedding_matrix = glove_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
# normalize the word embedding for fast histogram generating.
l2_norm = np.sqrt((embedding_matrix*embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
model.load_embedding_matrix(embedding_matrix)

hist_callback = mz.data_generator.callbacks.Histogram(embedding_matrix, bin_size=30, hist_mode='LCH')

pred_generator = mz.DataGenerator(valid_pack_processed, mode='point', callbacks=[hist_callback])
pred_x, pred_y = pred_generator[:]
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=32, model_save_path='DRMM_result_title_main_passage_keywords/DRMM_title_main_passage_keywords', once_every=1)

train_generator = mz.DataGenerator(
    train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1,
    batch_size=32,
    callbacks=[hist_callback]
)

history = model.fit_generator(train_generator, epochs=100, callbacks=[evaluate], verbose=0)