import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

#just building the useful plotting functions to plot graphs

def plot_revenue_trends(trends_data):
    df=pd.DataFrame(trends_data)
    fig,ax=plt.subplots()
    sns.lineplot(data=df,x='year',y='revenue',ax=ax)
    ax.set_title("Average Revenue Trends over Years")
    return fig 

def plot_correlation_heatmap(corr_data):
    df=pd.DataFrame(corr_data)
    fig,ax=plt.subplots()
    sns.heatmap(df,annot=True,cmap='coolwarm',ax=ax)
    ax.set_title("Feature Correlation")
    return fig 

def plot_genre_popularity(genre_data):
    fig,ax=plt.subplots(figsize=(12,6))
    genre_data.plot(ax=ax)
    ax.set_title("Genre Popularity over time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig 


