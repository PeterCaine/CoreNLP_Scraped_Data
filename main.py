from utils import data_in_fives, dataframe_constructor, dict_builder,\
    add_pre_processed_col, stanford_to_csv
from tqdm import tqdm
import pandas as pd
import glob
import pickle
import stanza
import os

def main():
#     uncomment below if error message stanza resources file not found download model again
#     stanza.download('en')
    nlp = stanza.Pipeline(lang='en',
                    processors='tokenize,mwt,pos,lemma, depparse, ner')
    prompt= "Enter path to folder with scraped pickle files: "
    pickle_path = input(prompt)

    for file in glob.glob(f'{pickle_path}/*.pkl'):
        basename = file.split('_')[-2].split('/')[-1]
        print(basename)
        f = pickle.load(open(file, 'rb'))
        list_o_five = data_in_fives(f)
        print(list_o_five)
        dict_out = dict_builder(list_o_five)
    #     print(dict_out)
        big_df = dataframe_constructor(dict_out)

        # concatenate list of dfs
        big_df.drop_duplicates(subset='text', inplace=True)
        big_df.reset_index(drop=True, inplace=True)
        try:
            big_df['joined_col'] = big_df.title.str.cat(big_df.text, sep=". ")
            dftext_to_list = list(big_df.joined_col)
        except:
            continue

        # preprocess with stanfordnlp (stanza)
        stanford_pp = []
        for text in tqdm(dftext_to_list):
            doc = nlp(text)
            stanford_pp.append(doc)

        # save stanford as CoNLL style tsv
        output = stanford_to_csv(stanford_pp)
        df_list = [pd.DataFrame.from_records(item) for item in output]
        output_df = pd.concat(df_list)
        output_df.reset_index(drop=True, inplace=True)
        if not os.path.exists('./airline_dataframe_dumps'):
            os.mkdir('./airline_dataframe_dumps')
            os.mkdir('./airline_dataframe_dumps/dataframe')
            os.mkdir('./airline_dataframe_dumps/stanford')
            os.mkdir('./airline_dataframe_dumps/conlls')
        output_df.to_csv(f'./airline_dataframe_dumps/conlls/{basename}.tsv', sep='\t', encoding='utf-8')

        # reconstruct dataframe with additional column
        df = add_pre_processed_col(stanford_pp, big_df, basename)
        df = df[['reviewer', 'review_date', 'rating', 'flight',
                'title', 'text', 'joined_col']]

        # dump df to pickle

        pickle.dump(df, open(f'./airline_dataframe_dumps/dataframe/{basename}_dataframe.pkl', 'wb'))
        pickle.dump(stanford_pp, open(f'./airline_dataframe_dumps/stanford/{basename}_stanforddoc.pkl', 'wb'))

if __name__ == "__main__":
    main()