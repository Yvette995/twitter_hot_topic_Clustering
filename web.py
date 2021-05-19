import streamlit as st
from function import *


st.sidebar.header('Twitter Hot Finder')
option = st.sidebar.selectbox('Choose',('input data','process data', 'show analyse'),index = 0)

uploaded_file = st.file_uploader('Please upload twitter data(.csv):')
if uploaded_file is not None:
    filepath =  'data.csv'
    eps_var = 0.6
    min_samples_var = 10

    @st.cache(allow_output_mutation=True)
    def main():
        df_non_outliers,rank_num,data_pca_tsne,label = process(filepath,eps_var,min_samples_var)
        return df_non_outliers,rank_num,data_pca_tsne,label
    df_non_outliers,rank_num,data_pca_tsne,label = main()

    if option == 'input data':
        st.write("Successfully Upload！")
        st.write(pd.read_csv(filepath,index_col=0)[:5000])

    if option == 'process data':
        st.write('The text has been transformed into vectors,and after the clustering process,twitter is divided into different categories')
        st.text('rank: The more popular the topic, the higher the ranking')
        st.write(df_non_outliers[['text','label','rank','pca_tsne']])
        st.write('The vectors of each hot topic are visualized as shown in the figure (different colors represent different categories)')
        fig = plt.figure()
        x = [i[0] for i in data_pca_tsne]
        y = [i[1] for i in data_pca_tsne]
        plt.scatter(x, y, c=label)
        plt.title('Hot Topic Vector')
        st.write(fig)


    if option == 'show analyse':
        st.write("Total",rank_num,"hot topics")
        rank_num = get_num_of_value_no_repeat(df_non_outliers['rank'])
        value = [df_non_outliers[df_non_outliers['rank'] == i].shape[0] for i in range(1, rank_num + 1)]
        yticks = [str(get_most_common_words(df_non_outliers[df_non_outliers['rank'] == i]['content_cut'],
                                                     top_n=5)) + str(i) for i in range(1, rank_num + 1)]


        fig = plt.figure(figsize=(13, 6), dpi=100)
        plt.subplot(122)
        ax = plt.gca()
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.invert_yaxis()
        plt.barh(range(1, rank_num + 1), value, align='center', linewidth=0)
        plt.yticks(range(1, rank_num + 1), yticks)
        for a, b in zip(value, range(1, rank_num + 1)):
            plt.text(a + 1, b, '%.0f' % a, ha='left', va='center')
        plt.title('Bar')
        st.write(fig)


        fig = plt.figure(figsize=(13, 6), dpi=100)
        plt.subplot(132)
        plt.pie(value, explode=[0.2] * rank_num, labels=yticks, autopct='%1.2f%%', pctdistance=0.7)
        plt.title('Pie')
        st.write(fig)

