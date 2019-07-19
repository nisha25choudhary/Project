from flask import Flask, request, render_template, url_for
from main_file import check_sms

app = Flask(__name__)

@app.route("/main")
def home():
    return render_template("index.html")

@app.route("/result",methods=["POST"])
def output():
    form_data = request.form["sms"]
    print(form_data)
    status = check_sms(form_data)
    return render_template("response.html",status=status)



if __name__ == "__main__":
    app.run(debug=True)
