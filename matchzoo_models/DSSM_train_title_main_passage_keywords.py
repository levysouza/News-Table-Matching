import matchzoo as mz
import pandas as pd
import numpy as np
import tensorflow as tf

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

preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=130, fixed_length_right=130, remove_stop_words=True)
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)

ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [mz.metrics.Precision()]

model = mz.models.DSSM()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = ranking_task
model.guess_and_fill_missing_params()
model.build()
model.compile()
model.backend.summary()

train_generator = mz.PairDataGenerator(train_processed, num_dup=1, num_neg=4, batch_size=32, shuffle=True)
valid_x, valid_y = valid_processed.unpack()

evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=32, model_save_path='DSSM_title_main_passage_keywords', once_every=1)

history = model.fit_generator(train_generator, epochs=100, callbacks=[evaluate], verbose=0)