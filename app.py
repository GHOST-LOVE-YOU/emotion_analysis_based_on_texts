import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from GoogleNews import GoogleNews
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import os
import datetime
from dotenv import load_dotenv
import requests

load_dotenv()

st.title('NLP系统展示:sunglasses:')
# ------------------------- 新闻分类 -------------------------
# 本来打算做推特帖子情感分析, 发现官方获取帖子的free API限制较多, 改为GoogleNews做新闻搜索
st.subheader("新闻搜索和情感分类")
def get_news(query):
    googlenews = GoogleNews(lang='en', region='US', period='1d')
    number_of_pages = 5
    final_list = []
    googlenews.search(query)
    print("Total Pages: ", googlenews.total_count())
    for page in range(1, number_of_pages + 1):
        page_result = googlenews.page_at(page)
        final_list = final_list + page_result
    return final_list

query = st.text_input("输入关键字")

if st.button("搜索"):
    with st.spinner("正在加载模型 ..."):
        classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    with st.spinner("正在加载最新新闻 ..."):
        allnews = get_news(query)
    with st.spinner("最新新闻已收到, 分析情绪中 ..."):
        df = pd.DataFrame(columns=["sentence", "date","best","second"])
        for curnews in allnews:
            cur_sentence = curnews["title"]
            cur_date = curnews["date"]
            model_outputs = classifier(cur_sentence)
            cur_result = model_outputs[0]
            # label 1
            label = cur_result[0]['label']
            score = cur_result[0]['score']
            percentage = round(score * 100, 2)
            str1 = label + " (" + str(percentage) + ")%"
            # label 2
            label = cur_result[1]['label']
            score = cur_result[1]['score']
            percentage = round(score * 100, 2)
            str2 = label + " (" + str(percentage) + ")%"
            df.loc[len(df.index)] = [cur_sentence, cur_date, str1, str2]
        st.dataframe(df)

        # 统计每个情绪标签的出现频次
        emotion_counts = df['best'].apply(lambda x: x.split(" ")[0]).value_counts()

        # 绘制柱状图
        st.subheader("情绪分析柱状图")
        fig, ax = plt.subplots()
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values, ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Frequency')
        plt.xlabel('Emotions')
        st.pyplot(fig)

        # 绘制饼状图
        st.subheader("情绪分析饼状图")
        fig, ax = plt.subplots()
        ax.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
# ----------------------------------------------------------------

# ------------------------- 论坛情感统计 -------------------------
# 数据来源byrbbs, 为了防止滥用, 不提供公用API, 原项目也会在验收后删除
st.subheader("论坛情感统计")

# Cache
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    return tokenizer, model

tokenizer, model_zh_en = load_model_and_tokenizer()

@st.cache_data(show_spinner=False)
def get_comments(time, max_comments=99999):
    url = os.getenv("API_URL") + f"?day={time}"
    token = os.getenv("API_TOKEN")
    headers = {"Authorization": f"token {token}"}
    response = requests.get(url, headers=headers)
    # 检查请求是否成功
    if response.status_code == 200:
        comments = response.json()
        df = pd.DataFrame(comments)
        df = df.head(max_comments)
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        st.error("Failed to load comments.")
        return None

@st.cache_data(show_spinner=False)
def batch_translate(batch_texts, _model, _tokenizer):
    inputs = _tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = _model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    batch_translations = [_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return batch_translations

def display_translations(comments, batch_size=16):
    if comments is not None:
        # 创建一个占位符
        placeholder = st.empty()

        for i in range(0, len(comments['content']), batch_size):
            batch_texts = comments['content'][i:i+batch_size].tolist()
            translations = batch_translate(batch_texts, model_zh_en, tokenizer)
            comments.loc[i:i+batch_size-1, 'content_en'] = translations
            # 使用占位符显示数据框
            placeholder.dataframe(comments[['content', 'content_en']][:i+batch_size])
        
        return comments[['content', 'content_en']]
    else:
        st.error("Failed to load comments.")


@st.cache_data(show_spinner=False)
def batch_sentiment_analysis(batch_texts, _classifier):
    model_outputs = _classifier(batch_texts)
    return model_outputs

def display_sentiments(translate_comments, batch_size=16):
    if translate_comments is not None:
        # 创建一个占位符
        placeholder = st.empty()

        df_sentiments = pd.DataFrame(columns=["content", "translation", "emotion", "score"])
        for i in range(0, len(translate_comments), batch_size):
            batch_texts = translate_comments['content_en'][i:i+batch_size].tolist()
            model_outputs = batch_sentiment_analysis(batch_texts, classifier)
            for j, output in enumerate(model_outputs):
                # 获取最高分数的情绪标签
                best_emotion = output[0]['label']
                best_score = output[0]['score']
                df_sentiments.loc[len(df_sentiments.index)] = [translate_comments['content'][i+j], translate_comments['content_en'][i+j], best_emotion, best_score]

            # 使用占位符显示数据框
            placeholder.dataframe(df_sentiments)

        # 统计每个情绪标签的出现频次
        emotion_counts = df_sentiments['emotion'].value_counts()

        # 绘制柱状图
        emotion_counts = df_sentiments['emotion'].value_counts()
        bar_fig = px.bar(emotion_counts, x=emotion_counts.index, y=emotion_counts.values,
                        labels={'index': 'Emotions', 'value': 'Frequency'},
                        title="论坛情感分析柱状图")
        st.plotly_chart(bar_fig, use_container_width=True)

        # 绘制饼状图
        pie_fig = px.pie(emotion_counts, names=emotion_counts.index, values=emotion_counts.values,
                        title="论坛情感分析饼状图", hole=.3)
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.error("Failed to analyze sentiments.")

# 选择日期
cur_date = datetime.datetime.now().strftime("%Y-%m-%d")
selected_date = st.date_input("选择日期", value=pd.to_datetime(cur_date))

settings = {
    "max_comments": 99999,
    "translate_batch_size": 16,
    "sentiment_batch_size": 16,
}

with st.sidebar:
    st.title("设置")
    st.header("最大获取帖子数")
    settings["max_comments"] = st.number_input("Max Comments", 1, 99999, 99999)

    st.header("翻译批处理大小")
    settings["translate_batch_size"] = st.number_input("Translate Batch Size", 1, 64, 16)

    st.header("情感分析批处理大小")
    settings["sentiment_batch_size"] = st.number_input("Sentiment Analysis Batch Size", 1, 64, 16)


if st.button("统计"):
    with st.spinner("正在加载模型 ..."):
        classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    with st.spinner("正在获取当天的帖子 ..."):
        comments = get_comments(selected_date, settings["max_comments"])
        st.dataframe(comments)
    with st.spinner("正在翻译帖子 ..."):
        translate_comments = display_translations(comments, settings["translate_batch_size"])
    with st.spinner("正在分析评论的情感倾向 ..."):
        display_sentiments(translate_comments, settings["sentiment_batch_size"])
