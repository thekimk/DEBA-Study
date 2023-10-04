from import_KK import *

def plot_wordcloud(df_wordfreq, title='Word Frequency',
                   background_color='white', max_font_size=200,
                   stopwords=None,
                   mask_location=None, mask_colorgen=True, max_words=100):
    # 세팅
    if mask_location != None:
        icon = Image.open(mask_location)
        mask = Image.new("RGB", icon.size, (255,255,255))
        mask.paste(icon,icon)
        mask = np.array(mask)
        #mask = np.array(Image.open(mask_location))
    else:
        mask = None
    if type(df_wordfreq) == pd.DataFrame:
        df_wordfreq = {row[0]: row[1] for _, row in df_wordfreq.iterrows()}
    
    # 시각화
    word_clouder = WordCloud(background_color=background_color,    # 배경색
                             contour_color='white',    # 경계색
                             contour_width=1,    # 경계두께
    #                          colormap='autumn',    # 글자컬러맵
                             width=1000, height=1000,    # 폭과 높이로 figsize랑 맞추어야
                             random_state=123,    # 랜덤 시각화 고정
    #                          prefer_horizontal=False,    # 수평글자로 기록
                             max_font_size=max_font_size,    # 최대 폰트 크기
                             max_words=max_words,    # 표현할 최대 단어 갯수)
                             stopwords=stopwords,
                             mask=mask,
                             font_path=FONT_PATHS[0])
    word_clouder = word_clouder.generate_from_frequencies(df_wordfreq)
    if mask_location != None and mask_colorgen:
        colormap = ImageColorGenerator(mask)
        word_clouder.recolor(color_func=colormap)
    else:
        pass
    plt.figure(figsize=(20,20))    # 캔버스 사이즈
    plt.imshow(word_clouder, interpolation='bilinear')
    plt.title(title, size=20)    # 제목과 사이즈
    plt.axis('off')    # 그래프 축을 제거
    plt.show()
        
    
def plot_bar_wordfreq(df_wordfreq, num_showkeyword = 100, num_subfigure = 5):
    # 하위함수 및 세팅
    def get_colordict(palette, number, start):
        pal = list(sns.color_palette(palette=palette, n_colors=number).as_hex())
        color = dict(enumerate(pal, start=start))
        return color
    
    # word, score(freq)만 포함인 경우
    if df_wordfreq.shape[1] == 2:
        df_wordfreq = df_wordfreq.sort_values(by=[df_wordfreq.columns[-1]], ascending=False)
        ## 데이터분리 인덱싱
        subindex = [[i[0],i[-1]+1] for i in np.array_split(range(num_showkeyword), num_subfigure)]
        fig, axs = plt.subplots(1, num_subfigure, figsize=(16,8), facecolor='white', squeeze=False)
        for col, idx in zip(range(0,num_subfigure), subindex):
            ## 데이터 및 시각화세팅
            df_sub = df_wordfreq[idx[0]:idx[-1]]
            x = list(df_sub.iloc[:,1])
            y = list(range(0,int(num_showkeyword/num_subfigure)))
            yticklabel = [word + ': ' + str(score) for word,score in zip(df_sub.iloc[:,0],df_sub.iloc[:,1])]
            score_max = df_wordfreq.iloc[:,1].max()
            ytickcolor = [get_colordict('viridis', score_max, 1).get(i) for i in df_sub.iloc[:,1]]
            ## barplot
            sns.barplot(x=x, y=y, data=df_sub, 
                        alpha=0.9, orient='h', palette=ytickcolor, ax=axs[0][col])
            axs[0][col].set_xlim(0, score_max+1)                     #set X axis range max
            axs[0][col].set_yticklabels(yticklabel, fontsize=12)
            axs[0][col].spines['bottom'].set_color('white')
            axs[0][col].spines['right'].set_color('white')
            axs[0][col].spines['top'].set_color('white')
            axs[0][col].spines['left'].set_color('white')    
        plt.tight_layout()    
        plt.show()
   
    # category, word, score(freq) 포함인 경우
    elif df_wordfreq.shape[1] == 3:
        df_wordfreq = df_wordfreq.sort_values(by=[df_wordfreq.columns[0], df_wordfreq.columns[-1]], ascending=[True, False])
        ## 데이터분리
        num_subfigure = len(df_wordfreq.iloc[:,0].unique())
        df_subs = [df_wordfreq[df_wordfreq.iloc[:,0] == i].iloc[:num_showkeyword,:] 
                   for i in df_wordfreq.iloc[:,0].unique()]
        fig, axs = plt.subplots(1, num_subfigure, figsize=(16,8), facecolor='white', squeeze=False)
        for col, df_sub in zip(range(0,num_subfigure), df_subs):
            ## 데이터 및 시각화세팅
            x = list(df_sub.iloc[:,2])
            y = list(range(0,df_sub.shape[0]))
            yticklabel = [word + ': ' + str(score) for word,score in zip(df_sub.iloc[:,1],df_sub.iloc[:,2])]
            score_max = df_wordfreq.iloc[:,2].max()
            ytickcolor = [get_colordict('viridis', score_max, 1).get(i) for i in df_sub.iloc[:,2]]
            ## barplot
            sns.barplot(x=x, y=y, data=df_sub)
            sns.barplot(x=x, y=y, data=df_sub, 
                        alpha=0.9, orient='h', palette=ytickcolor, ax=axs[0][col])
            axs[0][col].set_xlim(0, score_max+1)                     #set X axis range max
            axs[0][col].set_yticklabels(yticklabel, fontsize=12)
            axs[0][col].spines['bottom'].set_color('white')
            axs[0][col].spines['right'].set_color('white')
            axs[0][col].spines['top'].set_color('white')
            axs[0][col].spines['left'].set_color('white')    
            title = df_sub.iloc[:,0].unique()[0]
            axs[0][col].set_title(title)  
    
    
def plot_treemap_wordfreq(df_wordfreq, num_showkeyword=100, title='Treemap'):
    # word, score(freq)만 포함인 경우
    if df_wordfreq.shape[1] == 2:
        df_wordfreq = df_wordfreq.sort_values(by=[df_wordfreq.columns[-1]], ascending=False)
        fig = px.treemap(df_wordfreq[0:num_showkeyword],
                         path=[px.Constant(title), df_wordfreq.columns[0]],
                         values=df_wordfreq.columns[1],
                         color=df_wordfreq.columns[1],
                         color_continuous_scale='viridis',
                         color_continuous_midpoint=np.average(df_wordfreq.iloc[:,1])
                        )
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
        fig.show()
        
    # category, word, score(freq) 포함인 경우
    elif df_wordfreq.shape[1] == 3:
        df_wordfreq = df_wordfreq.sort_values(by=[df_wordfreq.columns[-1]], ascending=False)
        fig = px.treemap(df_wordfreq[:num_showkeyword], 
                         path=[px.Constant(title), df_wordfreq.columns[0], df_wordfreq.columns[1]],
                         values=df_wordfreq.columns[2],
                         color=df_wordfreq.columns[2],
                         hover_data=[df_wordfreq.columns[2]],
                         color_continuous_scale='viridis',
                         color_continuous_midpoint=np.average(df_wordfreq.iloc[:,2])
                        )
        fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
        fig.show()
    
    
def plot_donut_wordfreq(df_wordfreq, num_showkeyword=30):
    # 색상 설정
    pal = list(sns.color_palette(palette='Reds_r', n_colors=num_showkeyword).as_hex())

    # 시각화
    fig = px.pie(df_wordfreq[0:num_showkeyword], 
                 values=df_wordfreq.columns[1], names=df_wordfreq.columns[0],
                 color_discrete_sequence=pal)
    fig.update_traces(textposition='outside', textinfo='percent+label', 
                      hole=.6, hoverinfo="label+percent+name")
    fig.update_layout(width = 800, height = 600,
                      margin = dict(t=0, l=0, r=0, b=0))
    fig.show()
    
    
def plot_sunburst_wordfreq(df_wordfreq, title='Sunburst Plot'):
    # 라이브러리 및 하위 함수
    import plotly.graph_objects as go
    def get_colordict(palette,number,start):
        pal = list(sns.color_palette(palette=palette, n_colors=number).as_hex())
        color = dict(enumerate(pal, start=start))
        return color
    df_wordfreq = df_wordfreq.sort_values(by=[df_wordfreq.columns[0], df_wordfreq.columns[-1]], ascending=[True, False])
    
    # hierarchy 추가를 위한 정보 추가
    df_adding = df_wordfreq.groupby([df_wordfreq.columns[0]]).sum().reset_index()
    df_adding['temp'] = title
    df_adding = df_adding[['temp']+[df_wordfreq.columns[0]]+[df_wordfreq.columns[-1]]]
    df_adding.columns = df_wordfreq.columns
    
    # 정보 합치기 및 축정보 분리
    temp = pd.concat([df_wordfreq, df_adding], axis=0)
    sb_words = [str(i[-2])+'_'+str(i[-1]) if i[-2] in df_wordfreq.word.unique() else str(i[-2]) for i in temp.values]
    sb_score = list(temp.score)
    sb_category = list(temp.year)

    # 색상 생성
    ## color dict for words
    scores = list(df_wordfreq.score)
    nc = max(scores) - min(scores) + 1
    color_word = get_colordict('Reds', nc, min(scores))
    ## color dict for category
    scores = list(df_adding.score)
    nw = max(scores) - min(scores) + 1
    color_category = get_colordict('Reds', nw, min(scores))
    ## create color list
    colors = [color_word.get(i) for i in list(df_wordfreq.score)]+[color_category.get(i) for i in list(df_adding.score)]

    # sunburst
    fig = go.Figure(go.Sunburst(parents = sb_category,
                                labels = sb_words,
                                values = sb_score,
                                marker = dict(colors=colors)
                               ))
    fig.update_layout(width=800, height=800,
                      margin = dict(t=0, l=0, r=0, b=0))
    fig.show()
    
    
def plot_tsne_wordvec(df_wordvec, dim_reduction=3, num_showkeyword=100):
    # array 변환
    if df_wordvec.shape[1] == 1:
        X = np.array([row[0] for row in df_wordvec.values])
    elif df_wordvec.shape[1] == 2:
        X = np.array([row[0] for row in df_wordvec.iloc[:,1:].values])

    # 학습 및 차원변환
    tsne = TSNE(n_components=dim_reduction)
    w2v_tsne = tsne.fit_transform(X)
    if dim_reduction == 2:
        w2v_tsne = pd.DataFrame(w2v_tsne, index=list(df_wordvec.index), columns=['x', 'y'])
    elif dim_reduction == 3:
        w2v_tsne = pd.DataFrame(w2v_tsne, index=list(df_wordvec.index), columns=['x', 'y', 'z'])

    # 데이터 정리
    if df_wordvec.shape[1] == 2:
        w2v_tsne = pd.concat([w2v_tsne, df_wordvec.iloc[:,[0]]], axis=1)

    # t-SNE 시각화
    df_scatter = w2v_tsne.sample(num_showkeyword, random_state=123).copy()
    if dim_reduction == 2 and df_wordvec.shape[1] == 1:
        fig = px.scatter(df_scatter, x=df_scatter.columns[0], y=df_scatter.columns[1],
                         text=df_scatter.index)
    elif dim_reduction == 2 and df_wordvec.shape[1] == 2:
        fig = px.scatter(df_scatter, x=df_scatter.columns[0], y=df_scatter.columns[1],
                         color=df_scatter.columns[2],
                         text=df_scatter.index)
    elif dim_reduction == 3 and df_wordvec.shape[1] == 1:
        fig = px.scatter(df_scatter, x=df_scatter.columns[0], y=df_scatter.columns[1], z=df_scatter.columns[2],
                         text=df_scatter.index)
    elif dim_reduction == 3 and df_wordvec.shape[1] == 2:
        fig = px.scatter(df_scatter, x=df_scatter.columns[0], y=df_scatter.columns[1], z=df_scatter.columns[2],
                         color=df_scatter.columns[3],
                         text=df_scatter.index)
    fig.update_traces(textposition='bottom right')
    fig.update_layout(width = 1000, height = 1000)
    fig.show()