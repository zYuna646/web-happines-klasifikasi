from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

model = None  
accuracy = 0

@app.route('/')
def index():
    return render_template('train.html', accuracy=accuracy)

@app.route('/train', methods=['POST'])
def train_model():
    global model, accuracy

    uploaded_file = request.files['csv_file']
    if not uploaded_file:
        return redirect(url_for('index'))

    test_size = float(request.form['test_size']) 

    df = pd.read_csv(uploaded_file)


    train_data, test_data = train_test_split(df, test_size=test_size, random_state=12345)  

    model = DecisionTreeClassifier()
    model.fit(train_data[['infoavail', 'housecost', 'schoolquality', 'policetrust', 'streetquality', 'ëvents']], train_data['happy'])

    predictions = model.predict(test_data[['infoavail', 'housecost', 'schoolquality', 'policetrust', 'streetquality', 'ëvents']])

    accuracy = accuracy_score(test_data['happy'], predictions)
    accuracy = "{:.2f}".format(accuracy * 100)

    return redirect(url_for('index'))


@app.route('/classify', methods=['GET', 'POST'])
def classify_passenger():
    if request.method == 'POST':
        infoavail = int(request.form['infoavail'])
        housecost = int(request.form['housecost'])
        schoolquality = int(request.form['schoolquality'])
        policetrust = int(request.form['policetrust'])
        streetquality = int(request.form['streetquality'])
        events = int(request.form['events'])

        user_data = pd.DataFrame({'infoavail': [infoavail], 'housecost': [housecost], 'schoolquality': [schoolquality], 'policetrust': [policetrust], 'streetquality': [streetquality], 'ëvents': [events]})

        if model:
            prediction = model.predict(user_data)
            result = "Happy" if prediction[0] == 1 else "Not Happy" 
            return render_template('classify.html', result=result)
        else:
            return "Model belum dilatih. Silakan latih model terlebih dahulu."

    return render_template('classify.html')


if __name__ == '__main__':
    app.run(debug=True)


