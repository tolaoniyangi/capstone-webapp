from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, render_template, session, redirect
# import numpy as np
# import pandas as pd


# app = Flask(__name__)

# df = pd.DataFrame({'A': [0, 1, 2, 3, 4],
#                    'B': [5, 6, 7, 8, 9],
#                    'C': ['a', 'b', 'c--', 'd', 'e']})


# @app.route('/', methods=("POST", "GET"))
# def html_table():
#     return df.to_html(header="true", table_id="table")



# if __name__ == '__main__':
#     app.run(host='0.0.0.0')