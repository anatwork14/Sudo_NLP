from predict import predict

if __name__ == "__main__":
    # train_model() # train the model again if you first run / change in config.py
    
    text = "Dịch vụ không làm tôi hài lòng nhưng tôi sẽ quay lại hằng ngày vì đồ ăn ngon!"
    print("Input:", text)
    print("Prediction (SVM):", predict(text, model_choice="svm"))
    print("Prediction (Naive Bayes):", predict(text, model_choice="bayes"))
