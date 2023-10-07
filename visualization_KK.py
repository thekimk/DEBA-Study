from import_KK import *


### Date and Author: 20200820, Kyungwon Kim ###
### scaling of df
# feature_num_scaling(preprocessing.MinMaxScaler(), raw_stock[numeric_features])
def feature_num_scaling_df(scaler, df):
    print('Numerical Scaling...', '(', datetime.datetime.now(), ')')
    # fit
    scaler_fit = scaler.fit(df)
    
    # transform
    df_scaled = pd.DataFrame(scaler_fit.transform(df), 
                             index=df.index, columns=df.columns)
    
    return df_scaled


### Date and Author: 20190716, Kyungwon Kim ###
### Histgram Visualization
def plot_histogram(data_target, figsize=(10,5), fig_ncol=2):
    # Bold Expression
    start = '\033[1m'
    end = '\033[0m'
    
    # Counting the Number of Figures
    fig_num = 0
    for column in data_target.columns:
        if (data_target[column].dtype != 'object') & (data_target[column].dtype != 'datetime64[ns]') & (data_target[column].isnull().sum() != data_target.shape[0]):
            fig_num = fig_num + 1
            
    # Figure Size and Setting
    fig_nrow = math.ceil(fig_num/fig_ncol)
    fig_width = figsize[0] * fig_ncol
    fig_height = figsize[1] * (math.ceil(fig_num/fig_ncol))
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Each Histogram
    print(start + '<<< Histogram Summary >>>' + end)
    fig_num = 0
    for column in data_target.columns:
        ## Filtering (exclusion of 'object' and 'datetime64')
        if (data_target[column].dtype != 'object') & (data_target[column].dtype != 'datetime64[ns]') & (data_target[column].isnull().sum() != data_target.shape[0]):    
            fig_num = fig_num + 1
            fig.add_subplot(fig_nrow, fig_ncol, fig_num)
            x_min, x_max = np.max(abs(data_target[column]))*(-1), np.max(abs(data_target[column]))*1
#             x_hist = np.linspace(x_min, x_max, 20)
            data_hist = data_target[column][np.isfinite(data_target[column])]
            plt.hist(data_hist, bins='auto', density=False, histtype='bar', 
                     alpha=0.4, color='red', edgecolor='red', label='{}'.format(column))
            plt.legend(loc='upper right', framealpha=0.3, fancybox=True, fontsize=figsize[0]*fig_ncol)
            plt.title("Histogram (Incorrect Value: {} %)".format(data_target[column].isnull().sum() / len(data_target[column]) * 100), 
                      fontsize=figsize[0]*fig_ncol, fontname='Arial', fontweight='bold', 
                      fontstyle = 'italic', horizontalalignment='center')
            plt.ylabel("Count", fontsize=figsize[0]*fig_ncol, fontname='Arial', fontweight='bold', 
                       fontstyle = 'italic', horizontalalignment='center')
            plt.xlim(x_min, x_max)
    plt.tight_layout()
    plt.show()
  

### Date and Author: 20190716, Kyungwon Kim ###
### Timeplot Visualization
def plot_timeseries(data_target, figsize=(10,5), fig_ncol=2):
    # Bold Expression
    start = '\033[1m'
    end = '\033[0m'
    
    # Counting the Number of Figures
    fig_num = 0
    for column in data_target.columns:
        if (data_target[column].dtype != 'object') & (data_target[column].isnull().sum() != data_target.shape[0]):
            fig_num = fig_num + 1
            
    # Figure Size and Setting
    fig_nrow = math.ceil(fig_num/fig_ncol)
    fig_width = figsize[0] * fig_ncol
    fig_height = figsize[1] * (math.ceil(fig_num/fig_ncol))
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Each Timeplot
    print(start + '<<< Time Plot Summary >>>' + end)
    fig_num = 0
    for column in data_target.columns:
        ## Filtering
        if (data_target[column].dtype != 'object') & (data_target[column].isnull().sum() != data_target.shape[0]):  
            fig_num = fig_num + 1
            fig.add_subplot(fig_nrow, fig_ncol, fig_num)
            
            ### Target of Datetime
            if data_target[column].dtype == 'datetime64[ns]':
                data_target[column].groupby([data_target[column].dt.year]).count().plot(kind='bar')
#                 data_target[column].groupby([data_target[column].dt.year, data_target[column].dt.month]).count().plot(kind='bar')
                plt.title("Datetime Plot (Incorrect Value: {} %)".format(data_target[column].isnull().sum() / len(data_target[column]) * 100), 
                          fontsize=figsize[0]*fig_ncol, fontname='Arial', fontweight='bold', 
                          fontstyle = 'italic', horizontalalignment='center')
                plt.ylabel("Frequency", fontsize=figsize[0]*fig_ncol, fontname='Arial', fontweight='bold', 
                           fontstyle = 'italic', horizontalalignment='center')
        
            ### Others
            else: 
                plt.plot(data_target[column], alpha=0.4, color='red', linewidth=2, linestyle='--', label='{}'.format(column))
                plt.legend(loc='upper right', framealpha=0.3, fancybox=True, fontsize=figsize[0]*fig_ncol)
                plt.title("Time Plot (Incorrect Value: {} %)".format(data_target[column].isnull().sum() / len(data_target[column]) * 100), 
                          fontsize=figsize[0]*fig_ncol, fontname='Arial', fontweight='bold', 
                          fontstyle = 'italic', horizontalalignment='center')
                plt.ylabel("Value", fontsize=figsize[0]*fig_ncol, fontname='Arial', fontweight='bold', 
                           fontstyle = 'italic', horizontalalignment='center')
    plt.tight_layout()
    plt.show()
    

### Date and Author: 20200810, Kyungwon Kim ###
### Plots of DataFrame Values of time-series
def plot_timeseries_ver2(raw, save_local=False):
    # plot of index and return
    for sub in [i for i in raw.columns]:
        plt.figure(figsize=(16, 10))
        plt.plot(raw[sub], color='black', label=sub)
        plt.xticks(fontsize=25, rotation=0)
        plt.yticks(fontsize=25)
        if sub == raw.columns[len(raw.columns)-1]:
            plt.xlabel('Index', fontname='serif', fontsize=28)
        plt.ylabel(sub, fontname='serif', fontsize=28)
        plt.grid()
        if save_local:
            folder_location = os.path.join(os.getcwd(), 'Result', '')
            if not os.path.exists(folder_location):
                os.makedirs(folder_location)
            plt.savefig(folder_location+sub+'.pdf', dpi=600, bbox_inches='tight')
        plt.show()
        
        
### Date and Author: 20220220, Kyungwon Kim ###    
### Group Plots of DataFrame Origin Values of time-series
def plot_timeseries_dforigin(df, scaled=False, fontsize=20, ylabel='',
                             legend_colnum = 3, legend_anchor = (1.02,-0.1),
                             save_local=True, save_name_initial='PlotTime'):
    # scaling
    if scaled:
        df = feature_num_scaling_df(preprocessing.MinMaxScaler(), df)
        save_name_initial = save_name_initial + 'Scaled'
    
    # line plot
    if df.shape[1] == 1:
        df.plot(figsize=(16, 10), color='black')
    else:
        df.plot(figsize=(16, 10))
    plt.xticks(fontsize=fontsize, rotation=0)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Time', fontname='serif', fontsize=fontsize+4)
    plt.ylabel(ylabel, fontname='serif', fontsize=fontsize+4)
    plt.legend(fontsize=fontsize-2, ncol=legend_colnum, loc='best', bbox_to_anchor=legend_anchor)
    plt.grid(axis='y')
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Result', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        plt.savefig(os.path.join(folder_location, save_name_initial+'.pdf'), dpi=600, bbox_inches='tight')
    plt.show()

    
### Date and Author: 20220220, Kyungwon Kim ###
### Plots of Average and Standard Error of time-series DF
def plot_timeseries_dfmeanstd(df, scaled=False, fontsize=20, ylabel='Average',
                              save_local=True, save_name_initial='PlotTimeMeanStd'):
    # scaling
    if scaled:
        df = feature_num_scaling_df(preprocessing.MinMaxScaler(), df)
        save_name_initial = save_name_initial + 'Scaled'
        
    # calculate mean and std error
    def df_meanstd(df):
        df_mean = pd.DataFrame(df.mean(axis=1), columns=['Average'])
        df_upper = pd.DataFrame(df.mean(axis=1) + df.std(axis=1)/2, columns=['Standard Error(Upper)'])
        df_lower = pd.DataFrame(df.mean(axis=1) - df.std(axis=1)/2, columns=['Standard Error(Lower)'])
        
        return df_mean, df_upper, df_lower

    # plot 
    df_mean, df_upper, df_lower = df_meanstd(df)
    df_mean.plot(figsize=(16, 10), color='black', linewidth=2, legend=None)
    plt.fill_between(df_mean.index, np.ravel(df_upper), np.ravel(df_lower),
                     facecolor='gray', alpha=0.5)
    plt.xticks(fontsize=fontsize, rotation=0)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Time', fontname='serif', fontsize=fontsize+4)
    plt.ylabel(ylabel, fontname='serif', fontsize=fontsize+4)
    plt.grid(axis='y')
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Result', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        plt.savefig(os.path.join(folder_location, save_name_initial+'.pdf'), dpi=600, bbox_inches='tight')
    plt.show()
    
    
### Date and Author: 20220220, Kyungwon Kim ###    
### Group Plots of DataFrame Origin and Scaled Values of time-series
def plot_timeseries_df(df, fontsize=20, ylabel='Average',
                       legend_colnum = 3, legend_anchor = (1.02,-0.1),
                       save_local=True, save_name_initial='PlotTime'):
    # Origin Plot
    save_name_first = save_name_initial + 'Origin'
    scaled = False
    plot_timeseries_dforigin(df, scaled=scaled, fontsize=fontsize, ylabel='',
                             legend_colnum=legend_colnum, legend_anchor=legend_anchor,
                             save_local=save_local, save_name_initial=save_name_first)
    
    # Scaled Plot
    scaled = True
    plot_timeseries_dforigin(df, scaled=scaled, fontsize=fontsize, ylabel='',
                             legend_colnum=legend_colnum, legend_anchor=legend_anchor,
                             save_local=save_local, save_name_initial=save_name_first)
    
    # MeanStd Plot
    save_name_second = save_name_initial + 'OriginMeanStd'
    scaled = False
    plot_timeseries_dfmeanstd(df, scaled=scaled, fontsize=fontsize, ylabel=ylabel,
                              save_local=save_local, save_name_initial=save_name_second)
    
    # MeanStd Scaled Plot
    scaled = True
    plot_timeseries_dfmeanstd(df, scaled=scaled, fontsize=fontsize, ylabel=ylabel+' of Scaled Value',
                              save_local=save_local, save_name_initial=save_name_second)
    

### Date and Author: 20211030, Kyungwon Kim ###
### Plots of Average and Standard Error of Several time-series List DF list [DF1, DF2, ...]
def plot_timeseries_dfmeanstd_comparing(list_df, fontsize=20, xlabel='Time', ylabel='Average',
                                        list_legend = None, save_local=True, save_name='Plot_DFStat_Several'):
    # calculate mean and std error
    def df_meanstd(df):
        df_mean = pd.DataFrame(df.mean(axis=1), columns=['Average'])
        df_upper = pd.DataFrame(df.mean(axis=1) + df.std(axis=1)/2, columns=['Standard Error(Upper)'])
        df_lower = pd.DataFrame(df.mean(axis=1) - df.std(axis=1)/2, columns=['Standard Error(Lower)'])
        
        return df_mean, df_upper, df_lower
    
    # coloring
    import matplotlib.colors as mcolors
#     colors = list(mcolors.TABLEAU_COLORS.keys()) * list_df[0].shape[1]
    colors = ['tab:red', 'tab:blue', 'tab:pink', 'tab:cyan'] * list_df[0].shape[1]

    # plot 
    plt.figure(figsize=(16, 10))
    for idx, df in enumerate(list_df):
        df_mean, df_upper, df_lower = df_meanstd(df)
        plt.plot(df_mean, linewidth=2, color=colors[idx])
        plt.fill_between(df_mean.index, np.ravel(df_upper), np.ravel(df_lower),
                         facecolor=colors[idx], alpha=0.5)
        plt.xticks(fontsize=fontsize, rotation=0)
        plt.yticks(fontsize=fontsize)
        plt.xlabel(xlabel, fontname='serif', fontsize=fontsize+4)
        plt.ylabel(ylabel, fontname='serif', fontsize=fontsize+4)
        plt.grid(axis='x')
        plt.ylim(0, 7)
        plt.legend(list_legend, fontsize=fontsize, loc='upper left')
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Result', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        plt.savefig(folder_location+save_name+'.pdf', dpi=600, bbox_inches='tight')
    plt.show()
        
    
### Date and Author: 20200312, Kyungwon Kim ###
### Scatterplot Visualization
def plot_scatter(data_plot, data_axis, x_column, y_column, 
                 x_label='X', y_label='Y', legend_list=['ATL', 'BTL', 'SD'], 
                 color_list=['blue'], marker_list=['o'], annotation=False,
                 save_local=False, save_name=None):
    if type(data_plot) != list: data_plot = [data_plot].copy()
        
    # plot figure
    plt.figure(figsize=(14,6))
    for i, df in enumerate(data_plot):
        plt.scatter(x=x_column, y=y_column, s=200, data=df, color=color_list[i], marker=marker_list[i], alpha=0.4)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)
    x_range = np.max(data_axis[x_column]) - np.min(data_axis[x_column])
    y_range = np.max(data_axis[y_column]) - np.min(data_axis[y_column])
    plt.xlim(np.min(data_axis[x_column])-x_range*0.1, np.max(data_axis[x_column])+x_range*0.1)
    plt.ylim(np.min(data_axis[y_column])-y_range*0.1, np.max(data_axis[y_column])+y_range*0.1)
    
    # axis label
    plt.xlabel(x_label, fontname='serif', fontstyle='italic', fontsize=16)
    plt.ylabel(y_label, fontname='serif', fontstyle='italic', fontsize=16)
    
    # legend
    if len(legend_list) != 0:
        if type(legend_list) != list: legend_list = [legend_list].copy()
        plt.legend(legend_list, fontsize=13, loc=0, bbox_to_anchor=(1.11,1))
#         plt.legend(legend_list, fontsize=13, loc=0, bbox_to_anchor=((np.max(data_axis[x_column]))/10,1))

    # fill between layer
    plt.fill_between(x=np.array([np.mean(data_axis[x_column]), np.max(data_axis[x_column])+x_range*0.17]), 
                     y1=np.mean(data_axis[y_column]), y2=np.max(data_axis[y_column])+y_range*0.14,
                     color='gray', alpha=0.1)
    
    # average text
    plt.axhline(y=np.mean(data_axis[y_column]), linestyle='--', color='gray')
    plt.text(np.max(data_axis[x_column])+x_range*0.17, np.mean(data_axis[y_column]), 
             'Avg.={:.2f}%'.format(np.mean(data_axis[y_column])), 
             color='black', fontsize=13, style='italic', ha='center', va='center')
    plt.axvline(x=np.mean(data_axis[x_column]), linestyle='--', color='gray')
    plt.text(np.mean(data_axis[x_column]), np.max(data_axis[y_column])+y_range*0.14, 
             'Avg.={:.2f}%'.format(np.mean(data_axis[x_column])), 
             color='black', fontsize=13, style='italic', ha='center', va='center')
    
    # edge text
    plt.text(np.min(data_axis[x_column])-x_range*0.05, np.max(data_axis[y_column])+y_range*0.05, 'Optimize', 
             fontsize=11, ha='center', va='center')
    plt.text(np.min(data_axis[x_column])-x_range*0.05, np.min(data_axis[y_column])-y_range*0.05, 'Evaluate', 
             fontsize=11, ha='center', va='center')
    plt.text(np.max(data_axis[x_column])+x_range*0.05, np.max(data_axis[y_column])+y_range*0.05, 'Sustain', 
             fontsize=11, ha='center', va='center')
    plt.text(np.max(data_axis[x_column])+x_range*0.05, np.min(data_axis[y_column])-y_range*0.05, 'Expand', 
             fontsize=11, ha='center', va='center')
    
    # value lavel
    if annotation:
        for j, df in enumerate(data_plot):
            for i, (x_value, y_value) in enumerate(zip(df[x_column], df[y_column])):
                if (x_value >= np.mean(data_axis[x_column])) & (y_value >= np.mean(data_axis[y_column])):
                    plt.text(x_value, y_value, df[x_column].index[i], 
                             horizontalalignment='left', size=10, weight='bold')  
                elif (x_value >= np.mean(data_axis[x_column])) | (y_value >= np.mean(data_axis[y_column])):
                    plt.text(x_value, y_value, df[x_column].index[i], 
                             horizontalalignment='left', size=10)
                    
    # save file
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Result', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        plt.savefig(folder_location+save_name+'.pdf', dpi=600, bbox_inches='tight')
    plt.show()
    
    
### Date and Author: 20200810, Kyungwon Kim ###
### Plots of DataFrame Values ad Returns
def plot_valuereturn(raw, percent=False, save_local=False):
    # calculate returns of values
    if percent:
        raw_pct_change = feature_to_return_percent(raw)
    else:
        raw_pct_change = feature_to_return(raw)
        
    # plot of index and return
    for sub in [i for i in raw.columns]:
        plt.figure(figsize=(15,3))
        plt.subplots_adjust(wspace=0.05, hspace=0) # remove gabs between subplots
        ax1 = plt.subplot(121)
        ax1.plot(raw[sub], color='black', label=sub)
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
        if sub == raw.columns[len(raw.columns)-1]:
            plt.xlabel('Index', axes=ax1, fontname='serif', fontsize=16)
        plt.ylabel(sub, axes=ax1, fontname='serif', fontsize=16)
        ax1.grid()
        
        ax2 = plt.subplot(122)
        ax2.plot(raw_pct_change[sub], color='red', label=sub)
        ax2.xaxis.set_tick_params(labelsize=14)
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax2.yaxis.tick_right() # set the position of yaxis
        ax2.set_ylim([-np.max(abs(raw_pct_change[sub].values))*1.1, np.max(abs(raw_pct_change[sub].values))*1.1]) # set the range of yticks
        if sub == raw.columns[len(raw.columns)-1]:
            plt.xlabel('Index', axes=ax2, fontname='serif', fontsize=16)
        plt.ylabel('Daily returns', axes=ax2, fontname='serif', fontsize=16)
        ax2.yaxis.set_label_position('right') # set the position of ylabel
        ax2.grid()
        if save_local:
            folder_location = os.path.join(os.getcwd(), 'Result', '')
            if not os.path.exists(folder_location):
                os.makedirs(folder_location)
            plt.savefig(folder_location+sub+'.pdf', dpi=600, bbox_inches='tight')
        plt.show()
        
       
### Date and Author: 20211109, Kyungwon Kim ###
### Plots of Heatmap with Correlation
def plot_heatmap(df_square, title='Correlation', fontsize=20, 
                 save_local=True, save_name_initial=''):
    # heatmap of correlation
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(df_square, annot=False, cmap='BuPu', linewidths=.5, square=True, robust=True,
                cbar=True, cbar_kws=dict(shrink=0.82, orientation='vertical'))
    # plt.tick_params(labelbottom=False, labeltop=True)
    plt.title(title, fontsize=fontsize+2, weight='bold')
    plt.xticks(fontstyle='italic', fontsize=fontsize-6, weight='bold', rotation=90)
    plt.yticks(fontstyle='italic', fontsize=fontsize-6, weight='bold')
    plt.xlabel('', fontname='serif', fontsize=fontsize+4)
    plt.ylabel('', fontname='serif', fontsize=fontsize+4)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize-2)
    # cbar.ax.set_title(title, fontsize=fontsize-6, weight='bold')
    ## saving
    save_name = 'PlotHeatmap.pdf'
    if save_name_initial != '':
        save_name = save_name_initial + '_' + save_name
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Result', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        plt.savefig(os.path.join(folder_location, save_name), dpi=600, bbox_inches='tight')
    plt.show()
    
    
### Date and Author: 20211109, Kyungwon Kim ###
### Plots of Heatmap and it's Dendrogram from the DF
# 'ward', 'complete', 'average', 'single'
def plot_dendroheatmap(df_square, linkage='ward', distance_threshold=None, 
                       fontsize=20, title='Distance', save_local=True, save_name_initial=''):
    # dendrogram information
    corr_linkage = cluster.hierarchy.linkage(df_square, method=linkage)
    if distance_threshold == None:
        corr_dend = cluster.hierarchy.dendrogram(corr_linkage, orientation='left', 
                                                 labels=list(df_square),
                                                 no_plot=True)
    else:
        corr_dend = cluster.hierarchy.dendrogram(corr_linkage, orientation='left',
                                                 labels=list(df_square),
                                                 no_plot=True, color_threshold=distance_threshold)        
    
    # color making   
    ## 인접 그룹끼리 색상 재지정
#     num_color, num_unique = ['C1'], 1
#     for idx, c in enumerate(corr_dend['leaves_color_list'][1:]):
#         if c != corr_dend['leaves_color_list'][idx]:
#             num_unique = num_unique + 1
#         num_color.append('C'+str(num_unique))
#     corr_dend['leaves_color_list'] = num_color.copy()
    ## example
#     corr_dend['leaves_color_list'] = ['C1', 'C1', 'C2', 'C3', 'C3', 'C4', 'C5', 'C6', 'C6',
#                                        'C7', 'C7', 'C8', 'C8', 'C9', 'C9', 'C10', 'C10', 'C11',
#                                        'C12', 'C12', 'C13', 'C13', 'C14', 'C14', 'C15', 'C15', 'C15']
    ##
    columns_categ = pd.DataFrame(zip(pd.Series(corr_dend['ivl']), pd.Series(corr_dend['leaves_color_list'])))
    columns_categ.columns = ['Name', 'Cluster']
    columns_categ.set_index('Name', inplace=True)
    unique_categ = set(corr_dend['leaves_color_list'])
#     unique_color = sns.color_palette(n_colors=len(unique_categ))
    unique_color = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)]) for j in range(len(unique_categ))]
    cluster_color = dict(zip(unique_categ, unique_color))
    colors = columns_categ['Cluster'].map(cluster_color)
    colors.name = None

    # colorbar ticks
    dis_min = np.min(np.ravel(df_square))
    dis_max = np.max(np.ravel(df_square))
    ticks = [dis_min, 1/3*(dis_min + dis_max), 2/3*(dis_min + dis_max), dis_max]

    # plot of dendroheatmap
    fig = sns.clustermap(df_square, figsize=(16, 16),
                         annot=False, cmap='BuPu', linewidths=.5,
                         method=linkage, row_cluster=True, row_colors=colors, col_cluster=True, col_colors=colors,
                         cbar_kws={'orientation':'horizontal', 'ticks':ticks},
                         cbar_pos=[0.9, 0.05, 0.25, 0.05])
    fig.ax_heatmap.set_xticklabels(fig.ax_heatmap.get_xmajorticklabels(), 
                                   fontstyle='italic', fontsize=fontsize, weight='bold')
    ## xtick color
    for idx, val in enumerate(fig.ax_heatmap.get_xticklabels()):
        val_color = colors[str(val).split(", '")[1][:-2]]
        val.set_color(val_color)
    fig.ax_heatmap.set_yticklabels(fig.ax_heatmap.get_ymajorticklabels(), 
                                   fontstyle='italic', fontsize=fontsize, weight='bold')
    ## ytick color
    for idx, val in enumerate(fig.ax_heatmap.get_yticklabels()):
        val_color = colors[str(val).split(", '")[1][:-2]]
        val.set_color(val_color)
    fig.ax_cbar.set_title(title, fontsize=fontsize+2, weight='bold')
    fig.ax_cbar.tick_params(labelsize=fontsize)
    ## saving
    save_name = 'PlotDendtoHeatmap.pdf'
    if save_name_initial != '':
        save_name = save_name_initial + '_' + save_name
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Result', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        plt.savefig(os.path.join(folder_location, save_name), dpi=600, bbox_inches='tight')
    plt.show()
    
    return colors


### Date and Author: 20211109, Kyungwon Kim ###
### Plots of TimeSeries by the Cluster of Dendrogram
def plot_dendrots(df_square, cluster_color, save_local=True, save_name_initial=''):
    # rearrange of cluster color
    df_cluster = pd.DataFrame(cluster_color, columns=['Color']).reset_index()
    def str_cat(x):
        return x.str.cat(sep=", ").split(', ')
    df_cluster = df_cluster.groupby(df_cluster['Color']).agg({'Name': str_cat})
    df_cluster = df_cluster.loc[cluster_color.unique(),:]

    # subplots by cluster
    fig, axes = plt.subplots(nrows=df_cluster.shape[0], ncols=1, figsize=(5, 10))
    for row in range(df_cluster.shape[0]):
        df_cluster_sub = df_square[df_cluster.iloc[row,0]].reset_index().iloc[:,1:]
        df_cluster_sub.plot(ax=axes[row], color=df_cluster.index[row], legend=None)
        for pos in ['top', 'bottom', 'right', 'left']:
            axes[row].spines[pos].set_color(df_cluster.index[row])
#         axes[row].axis('off')
        axes[row].set_xticks([])
        axes[row].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0.02)
    ## saving
    save_name = 'PlotDendtoTimeSeries.pdf'
    if save_name_initial != '':
        save_name = save_name_initial + '_' + save_name
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Result', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        plt.savefig(os.path.join(folder_location, save_name), dpi=600, bbox_inches='tight')
    plt.show()
    
    
### Date and Author: 20220305, Kyungwon Kim ###
### Plots of Heatmap, Dendrogram, and TS by the Cluster from the DF
def plot_dendroheatmapts(df_square, linkage='ward', distance_threshold=None, 
                         fontsize=20, title='Distance',
                         save_local=True, save_name_initial=''):
    # dendro heatmap
    cluster_color = plot_dendroheatmap(df_square, linkage=linkage, distance_threshold=distance_threshold,
                                       fontsize=fontsize, title=title, 
                                       save_local=save_local, save_name_initial=save_name_initial)
    # time plot for each cluster
    plot_dendrots(df_square, cluster_color, save_name_initial=save_name_initial)
    
    return cluster_color


### Date and Author: 20230425, Kyungwon Kim ###
### 클래스 별 통계량 비교 시각화
def plot_classfrequency(df, Y_colname, label_list=['0', '1']):
    # 클래스 별 데이터 분리 및 통계량 연산
    X_colname = [x for x in df.columns if x not in Y_colname]
    stat_compare = []
    for col in X_colname:
        label0 = df[df[Y_colname[0]]==0][col]
        label1 = df[df[Y_colname[0]]==1][col]
        stat_compare.append([label0.mean(), label0.median(), label0.std(), 0])
        stat_compare.append([label1.mean(), label1.median(), label1.std(), 1])
    stat_compare = pd.DataFrame(stat_compare)
    stat_compare.columns = ['Mean', 'Median', 'Standard Deviation', 'Y Label']

    # 통계량 시각화
    for col in stat_compare.columns[:-1]:
        plt.figure(figsize=(10,5))
        sns.histplot(data=stat_compare[stat_compare[col] <= 100], x=col, hue='Y Label', 
                     stat='frequency', kde=False, multiple='fill')
        plt.xlabel(xlabel=col, fontsize=16)
        plt.ylabel(ylabel='Frequency', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(label_list, fontsize=14)
        plt.show()
        plt.figure(figsize=(10,5))
        sns.histplot(data=stat_compare[stat_compare[col] <= 100], x=col, hue='Y Label', 
                     stat='frequency', kde=True, multiple='dodge')
        plt.xlabel(xlabel=col, fontsize=16)
        plt.ylabel(ylabel='Frequency', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(label_list, fontsize=14)
        plt.show()
        

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
        
    
def plot_bar_wordfreq(df_wordfreq, num_showkeyword=100, num_subfigure=5, title='Bar Plot'):
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
        ## 통합차트 시각화
        fig = px.bar(df_wordfreq, x=df_wordfreq.columns[1], y=df_wordfreq.columns[2], color=df_wordfreq.columns[0])
        fig.update_layout(
            title=dict(text='Ageism'),
            font=dict(size=12)
        )
        fig.show()
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
    
    
# plot_tsne_wordvec(word_vec_w2v, dim_reduction=3, num_showkeyword=1000)
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
        fig = px.scatter_3d(df_scatter, x=df_scatter.columns[0], y=df_scatter.columns[1], z=df_scatter.columns[2],
                            text=df_scatter.index)
    elif dim_reduction == 3 and df_wordvec.shape[1] == 2:
        fig = px.scatter_3d(df_scatter, x=df_scatter.columns[0], y=df_scatter.columns[1], z=df_scatter.columns[2],
                            color=df_scatter.columns[3],
                            text=df_scatter.index)
    fig.update_traces(textposition='bottom right')
    fig.update_layout(width = 1000, height = 1000)
    fig.show()
    

# measure change: https://chaelist.github.io/docs/network_analysis/social_network/
# https://graphsandnetworks.com/community-detection-using-networkx/
def plot_networkx(df_freq, df_pairweight, filter_criteria=None, plot=True, node_size='score', edge_label=False):
    # 빈 그래프 생성
    G = nx.Graph()   # undirected graph
    
    # 입력값 필터 & node+edge 값 반영
    display('Descriptive statistics of pairweight: ', df_pairweight.describe().T)
    if filter_criteria == None and np.mean(df_pairweight.iloc[:,-1]) >= 0:
        filter_criteria = np.mean(df_pairweight.iloc[:,-1])*2
    elif filter_criteria == None and np.mean(df_pairweight.iloc[:,-1]) < 0:
        filter_criteria = abs(np.mean(df_pairweight.iloc[:,-1]))*2
    df_pair = df_pairweight[df_pairweight.iloc[:,-1] > filter_criteria]    # weight 필터
    if df_pairweight.shape[1] == 3:
        G.add_edges_from([(each[-3], each[-2], {'weight': each[-1]}) 
                          for each in df_pair.values])    # edge & attributes 반영
        G.add_nodes_from([(each[-2], {'score': each[-1]}) 
                          for each in df_freq.values])    # node & attributes 반영
        word_filter = df_pair.iloc[:,:2].stack().tolist()    # unique word 추출
    elif df_pairweight.shape[1] == 4:
        G.add_edges_from([(each[-3], each[-2], {'weight': each[-1], 'group_edge': each[-4]}) 
                          for each in df_pair.values])    # edge & attributes 반영
        G.add_nodes_from([(each[-2], {'score': each[-1], 'group_node': each[-3]}) 
                          for each in df_freq.values])    # node & attributes 반영
        word_filter = df_pair.iloc[:,1:3].stack().tolist()    # unique word 추출
    df_freq = df_freq[df_freq.word.apply(lambda x: True if x in word_filter else False)]    # word-score에서 unique word 필터  
#     display(G.nodes(data=True), G.edges(data=True))    # 입력 확인
    
    # 통계량 추출
    degree_centrality = sorted(nx.degree_centrality(G).items(), key=lambda x:x[1], reverse=True)
    betweenness_centrality = sorted(nx.betweenness_centrality(G).items(), key=lambda x:x[1], reverse=True)
    closeness_centrality = sorted(nx.closeness_centrality(G).items(), key=lambda x:x[1], reverse=True)
    eigenvector_centrality = sorted(nx.eigenvector_centrality(G).items(), key=lambda x:x[1], reverse=True)
    pagerank = sorted(nx.pagerank(G).items(), key=lambda x:x[1], reverse=True)
    ## node attributes 반영
    for i in range(len(pagerank)):
        G.add_node(degree_centrality[i][0], degree_centrality=degree_centrality[i][1])
        G.add_node(betweenness_centrality[i][0], betweenness_centrality=betweenness_centrality[i][1])
        G.add_node(closeness_centrality[i][0], closeness_centrality=closeness_centrality[i][1])
        G.add_node(eigenvector_centrality[i][0], eigenvector_centrality=eigenvector_centrality[i][1])
        G.add_node(pagerank[i][0], pagerank=pagerank[i][1])
    ## 정리
    centrality = pd.concat([pd.DataFrame(degree_centrality), pd.DataFrame(betweenness_centrality), 
                            pd.DataFrame(closeness_centrality), pd.DataFrame(eigenvector_centrality),
                            pd.DataFrame(pagerank)], axis=1)
    centrality.columns = ['word', 'degree', 'word', 'betweenness', 'word', 'closeness', 'word', 'eigenvector', 'word', 'pagerank']

    # 레이아웃 반복
    if plot:
        ## 하위함수
        def rescale(data_list,newmin,newmax):
            arr = list(data_list)
            return [(x-min(arr))/(max(arr)-min(arr))*(newmax-newmin)+newmin for x in arr]
        ## degree에 따른 중심 그래프 생성
        k_core1 = np.quantile([val for key, val in G.degree()], 0.5)
        G_core1 = nx.k_core(G, k=k_core1)
        ## node size 결정
        if node_size == 'score':
            node_size = list(nx.get_node_attributes(G, 'score').values())
            node_size_core1 = list(nx.get_node_attributes(G_core1, 'score').values())
        elif node_size == 'degree':
            node_size = list(nx.get_node_attributes(G, 'degree_centrality').values())
            node_size_core1 = list(nx.get_node_attributes(G_core1, 'degree_centrality').values())
        elif node_size == 'eigenvector':
            node_size = list(nx.get_node_attributes(G, 'eigenvector_centrality').values())
            node_size_core1 = list(nx.get_node_attributes(G_core1, 'eigenvector_centrality').values())
        elif node_size == 'pagerank':
            node_size = list(nx.get_node_attributes(G, 'pagerank').values())
            node_size_core1 = list(nx.get_node_attributes(G_core1, 'pagerank').values())
        node_size, node_size_core1 = rescale(node_size, 500, 2500), rescale(node_size_core1, 1000, 5000)
        ## node color 결정(Gradients 필요시)
        node_color = [mpl.cm.get_cmap('Oranges')(i) for i in node_size]
        node_color_core1 = [mpl.cm.get_cmap('Reds')(i) for i in node_size_core1]
        ## edge weight 결정
        edge_weight = rescale([float(G[u][v]['weight']) for u,v in G.edges],0.5,2)
        edge_weight_core1 = rescale([float(G[u][v]['weight']) for u,v in G.edges],2.5,5)
        ## 레이어 종류 별 시각화
#         pos = nx.bipartite_layout(G)    # 양분그리기는 별도 라벨 필요
        for idx, layout in enumerate([nx.kamada_kawai_layout(G, weight=None), 
                                      nx.circular_layout(G), 
                                      nx.fruchterman_reingold_layout(G)]):
            plt.figure(figsize=(16,10))
            plt.style.use('dark_background')
            nx.draw_networkx(G, with_labels=True, alpha=0.2, 
                             node_size=node_size, node_color='Yellow', 
                             width=edge_weight, edge_color='red', font_family=FONT_NAME, pos=layout)
            nx.draw_networkx_labels(G, font_size=9, font_color='white', font_weight='bold', alpha=0.4, 
                                    font_family=FONT_NAME, pos=layout)
            nx.draw_networkx(G_core1, with_labels=True, alpha=0.5, 
                             node_size=node_size_core1, node_color='red', 
                             width=edge_weight_core1, edge_color='red', font_family=FONT_NAME, pos=layout)
            nx.draw_networkx_labels(G_core1, font_size=12, font_color='white', font_weight='bold', alpha=1,
                                    font_family=FONT_NAME, pos=layout)
            if edge_label:
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, edge_labels=edge_labels, pos=layout)
            plt.show()
        plt.style.use('default')
    
    return G, centrality




