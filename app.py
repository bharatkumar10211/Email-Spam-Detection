from flask import Flask,render_template,request
from spam import predict_message

app=Flask(__name__)
@app.route("/",methods=["GET","POST"])
def home():
    result=spam_prob=ham_prob=None
    msg=""
    if request.method=="POST":
        msg=request.form["message"]
        result,spam_prob,ham_prob=predict_message(msg)
    return render_template("index.html",result=result,spam_prob=spam_prob,ham_prob=ham_prob,messag=msg)

if __name__=="__main__":
    app.run(debug=True)