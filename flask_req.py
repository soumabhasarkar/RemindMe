from flask import Flask
from flask import request


app = Flask(__name__)

@app.route('/get', methods=['GET', 'POST'])
def text_category():
    print("Req Data")
    print(request.get_data())
    return ("how are you today?")

if __name__=="__main__":
    app.run()
