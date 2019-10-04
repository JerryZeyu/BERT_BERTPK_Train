import os
import pandas as pd
from collections import OrderedDict
from spacy.lang.en import English
import warnings
nlp = English()
def loadLookupLemmatizer(file_name):

    lemmatizerHashmap={}
    with open(file_name, 'r+') as f:
        text = f.readlines()
        for line in text:
            line=line.replace("[\\s]+", "\t").strip()
            split_list = line.lower().split("\t")
            lemma = split_list[0].strip().lower()
            word = split_list[1].strip().lower()
            lemmatizerHashmap[word] = lemma

    return lemmatizerHashmap
def _read_tsv(input_file):
    header = []
    uid = None
    df = pd.read_csv(input_file, sep='\t')
    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        else:
            header.append(name)
    if not uid or len(df) == 0:
        warnings.warn('Possibly misformatted file: '+input_file)
        return []
    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isnull(s))), 1).tolist()
def _read_tsv_DEP(input_file):
    header = []
    uid = None
    df = pd.read_csv(input_file, sep='\t')
    if '[SKIP] DEP' in df.columns:
        df_filter = df[df['[SKIP] DEP'].notna()].copy()
        for name in df_filter.columns:
            if name.startswith('[SKIP]'):
                if 'UID' in name and not uid:
                    uid = name
            else:
                header.append(name)
        if not uid or len(df_filter) == 0:
            warnings.warn('Possibly misformatted file: ' + input_file)
            return []
        return df_filter.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isnull(s))), 1).tolist()
    else:
        warnings.warn('Possibly not contain DEP: ' + input_file)
        return []
def _read_tsv_normal_words(input_file):
    header = []
    uid = None
    df = pd.read_csv(input_file, sep='\t')
    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        elif name.startswith('[FILL]'):
            continue
        else:
            header.append(name)
    if not uid or len(df) == 0:
        warnings.warn('Possibly misformatted file: '+input_file)
        return []
    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isnull(s))), 1).tolist()

def explanations_filtering(dict_explanations, dict_explanations_normal, dict_explanations_DEP, train_data_dir, dev_data_dir):
    explanations_id_list = []
    df_q_train = pd.read_csv(train_data_dir, sep='\t')
    df_q_dev = pd.read_csv(dev_data_dir, sep='\t')
    for _, row in df_q_train.iterrows():
        if 'SUCCESS' not in str(row['flags']).split(' '):
            continue
        for single_row_id in list(OrderedDict.fromkeys(str(row['explanation']).split(' ')).keys()):
            explanations_id_list.append(single_row_id.split('|')[0])
    for _, row in df_q_dev.iterrows():
        if 'SUCCESS' not in str(row['flags']).split(' '):
            continue
        for single_row_id in list(OrderedDict.fromkeys(str(row['explanation']).split(' ')).keys()):
            explanations_id_list.append(single_row_id.split('|')[0])
    removed_list = [item for item in dict_explanations_DEP if item not in explanations_id_list]
    print('dict explanations lenth: ', len(dict_explanations.keys()))
    print('dict explanations normal lenth: ', len(dict_explanations_normal.keys()))
    print('remove list length: ', len(removed_list))
    print('set remove list lenth: ', len(set(removed_list)))
    for id in removed_list:
        del dict_explanations[id]
        del dict_explanations_normal[id]
    print('filtered dict explanations lenth: ', len(dict_explanations.keys()))
    print('filtered dict explanations normal lenth: ', len(dict_explanations_normal.keys()))
    return dict_explanations, dict_explanations_normal
def read_tables(data_dir, train_data_dir, dev_data_dir):
    explanations = []
    explanations_DEP = []
    explanations_normal = []
    for path, _, files in os.walk(
            os.path.join(data_dir, 'tables')):
        for file in files:
            explanations += _read_tsv(os.path.join(path, file))
            explanations_normal += _read_tsv_normal_words(os.path.join(path, file))
            explanations_DEP += _read_tsv_DEP(os.path.join(path, file))
    if not explanations:
        warnings.warn('Empty explanations')
    dict_explanations = {}
    dict_explanations_normal = {}
    dict_explanations_DEP = {}
    for item in explanations:
        dict_explanations[item[0]] = item[1]
    #print(dict_explanations['7471-bdd9-0d81-87f4'])
    for item_ in explanations_normal:
        dict_explanations_normal[item_[0]] = item_[1]
    for item__ in explanations_DEP:
        dict_explanations_DEP[item__[0]] = item__[1]
    filtered_dict_explanations, filtered_dict_explanations_normal = explanations_filtering(dict_explanations, dict_explanations_normal,
                                                                                           dict_explanations_DEP,
                                                                                           train_data_dir, dev_data_dir)
    return filtered_dict_explanations, filtered_dict_explanations_normal
    #print(dict_explanations['7471-bdd9-0d81-87f4'])
    #return dict_explanations, explanations_normal

def remove_stopWord(dict_explanations):
    dict_stopwords={}
    for id, table_row in dict_explanations.items():
        parsed_table_row = nlp(table_row)
        removed_table_row = [token.text for token in parsed_table_row if not token.is_stop]
        dict_stopwords[id] = removed_table_row
    return dict_stopwords

