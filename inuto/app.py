from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

df_grammar = pd.read_csv('inuto/datasets/grammar.csv')
df_listening = pd.read_csv('inuto/datasets/listening.csv')
df_reading = pd.read_csv('inuto/datasets/reading.csv')
df_word = pd.read_csv('inuto/datasets/word.csv')

# コサイン類似度の定義
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@app.route('/')
def index():
    return render_template('index.html')

# 類似度計算
def recommended_word_books(values):
    input_data = values  # ユーザーのアンケート回答に基づく数値リストを使用
    result = []
    for i in range(len(df_word)):
        data = df_word.iloc[i, 1:].values
        data = data.astype(np.float32)
        result.append(cos_sim(input_data, data))

    # ソートされた類似度のインデックスを取得
    if len(result) > 0:
        max_indices = np.argsort(result)[::-1]  # 類似度を降順にソートしたインデックスを取得
        top_indices = max_indices[:3]  # 上位3つのインデックスを取得
    else:
        top_indices = []

    recommended_books = [df_word.iloc[i, 0] for i in top_indices]
    top_scores = [result[i] for i in top_indices]
    return recommended_books, top_scores

@app.route('/word_radio', methods=['GET', 'POST'])
def word_radio():
    if request.method == 'GET':
        return render_template('word_radio.html')
    elif request.method == 'POST':
        answer_sum1 = int(request.form['answer1']) + int(request.form['answer2']) + int(request.form['answer3']) + int(request.form['answer4'])
        answer_sum2 = int(request.form['answer5']) + int(request.form['answer6'])
        answer_sum3 = int(request.form['answer7']) + int(request.form['answer8'])
        answer_sum4 = int(request.form['answer9']) + int(request.form['answer10'])
        answer_sum5 = int(request.form['answer11'])

        values_list = [answer_sum1, answer_sum2, answer_sum3, answer_sum4, answer_sum5]
        recommended_books_list, top_scores = recommended_word_books(values_list)
        total_value = sum(values_list)

        data_path = []
        for book_name in recommended_books_list:
            image_file_name = book_name + ".jpg"
            image_file_path = f"./static/word_picture/{image_file_name}"
            data_path.append(image_file_path)

        return render_template('result.html', data_path=data_path, total_value=total_value, result=top_scores)

# 類似度計算
def recommended_grammar_books(values):
    input_data = values  # ユーザーのアンケート回答に基づく数値リストを使用
    result = []
    for i in range(len(df_grammar)):
        data = df_grammar.iloc[i, 1:].values
        data = data.astype(np.float32)
        result.append(cos_sim(input_data, data))

    # ソートされた類似度のインデックスを取得
    if len(result) > 0:
        max_indices = np.argsort(result)[::-1]  # 類似度を降順にソートしたインデックスを取得
        top_indices = max_indices[:3]  # 上位3つのインデックスを取得
    else:
        top_indices = []

    recommended_books = [df_grammar.iloc[i, 0] for i in top_indices]
    top_scores = [result[i] for i in top_indices]
    return recommended_books, top_scores

@app.route('/grammar_radio', methods=['GET', 'POST'])
def grammar_radio():
    if request.method == 'GET':
        return render_template('grammar_radio.html')
    elif request.method == 'POST':
        answer_sum1 = int(request.form['answer1']) + int(request.form['answer2']) + int(request.form['answer3']) + int(request.form['answer4'])
        answer_sum2 = int(request.form['answer5']) + int(request.form['answer6'])
        answer_sum3 = int(request.form['answer7']) + int(request.form['answer8'])
        answer_sum4 = int(request.form['answer9']) + int(request.form['answer10'])
        answer_sum5 = int(request.form['answer11']) + int(request.form['answer12'])

        values_list = [answer_sum1, answer_sum2, answer_sum3, answer_sum4, answer_sum5]
        recommended_books_list, top_scores = recommended_grammar_books(values_list)
        total_value = sum(values_list)

        data_path = []
        for book_name in recommended_books_list:
            image_file_name = book_name + ".jpg"
            image_file_path = f"./static/grammar_picture/{image_file_name}"
            data_path.append(image_file_path)

        return render_template('result.html', data_path=data_path, total_value=total_value, result=top_scores)
   
# 類似度計算
def recommended_reading_books(values):
    input_data = values  # ユーザーのアンケート回答に基づく数値リストを使用
    result = []
    for i in range(len(df_reading)):
        data = df_reading.iloc[i, 1:].values
        data = data.astype(np.float32)
        result.append(cos_sim(input_data, data))

    # ソートされた類似度のインデックスを取得
    if len(result) > 0:
        max_indices = np.argsort(result)[::-1]  # 類似度を降順にソートしたインデックスを取得
        top_indices = max_indices[:3]  # 上位3つのインデックスを取得
    else:
        top_indices = []

    recommended_books = [df_reading.iloc[i, 0] for i in top_indices]
    top_scores = [result[i] for i in top_indices]
    return recommended_books, top_scores

@app.route('/reading_radio', methods=['GET', 'POST'])
def reading_radio():
    if request.method == 'GET':
        return render_template('reading_radio.html')
    elif request.method == 'POST':
        answer_sum1 = int(request.form['answer1']) + int(request.form['answer2']) + int(request.form['answer3']) + int(request.form['answer4'])
        answer_sum2 = int(request.form['answer5']) + int(request.form['answer6'])
        answer_sum3 = int(request.form['answer7']) + int(request.form['answer8'])
        answer_sum4 = int(request.form['answer9']) + int(request.form['answer10'])
        answer_sum5 = int(request.form['answer11']) + int(request.form['answer12'])

        values_list = [answer_sum1, answer_sum2, answer_sum3, answer_sum4, answer_sum5]
        recommended_books_list, top_scores = recommended_reading_books(values_list)
        total_value = sum(values_list)

        data_path = []
        for book_name in recommended_books_list:
            image_file_name = book_name + ".jpg"
            image_file_path = f"./static/reading_picture/{image_file_name}"
            data_path.append(image_file_path)

        return render_template('result.html', data_path=data_path, total_value=total_value, result=top_scores)

# 類似度計算
def recommended_listening_books(values):
    input_data = values  # ユーザーのアンケート回答に基づく数値リストを使用
    result = []
    for i in range(len(df_listening)):
        data = df_listening.iloc[i, 1:].values
        data = data.astype(np.float32)
        result.append(cos_sim(input_data, data))

    # ソートされた類似度のインデックスを取得
    if len(result) > 0:
        max_indices = np.argsort(result)[::-1]  # 類似度を降順にソートしたインデックスを取得
        top_indices = max_indices[:3]  # 上位3つのインデックスを取得
    else:
        top_indices = []

    recommended_books = [df_listening.iloc[i, 0] for i in top_indices]
    top_scores = [result[i] for i in top_indices]
    return recommended_books, top_scores

@app.route('/listening_radio', methods=['GET', 'POST'])
def listening_radio():
    if request.method == 'GET':
        return render_template('listening_radio.html')
    elif request.method == 'POST':
        answer_sum1 = int(request.form['answer1']) + int(request.form['answer2']) + int(request.form['answer3']) + int(request.form['answer4'])
        answer_sum2 = int(request.form['answer5']) + int(request.form['answer6'])
        answer_sum3 = int(request.form['answer7']) + int(request.form['answer8'])
        answer_sum4 = int(request.form['answer9']) + int(request.form['answer10'])

        values_list = [answer_sum1, answer_sum2, answer_sum3, answer_sum4]
        recommended_books_list, top_scores = recommended_listening_books(values_list)
        total_value = sum(values_list)

        data_path = []
        for book_name in recommended_books_list:
            image_file_name = book_name + ".jpg"
            image_file_path = f"./static/listening_picture/{image_file_name}"
            data_path.append(image_file_path)

        return render_template('result.html', data_path=data_path, total_value=total_value, result=top_scores)

if __name__ == '__main__':
    app.run(debug=True)