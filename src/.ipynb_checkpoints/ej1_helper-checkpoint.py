from keras import backend as K

# Precision and Recall metrics implementation based from https://stackoverflow.com/questions/42606207/keras-custom-decision-threshold-for-precision-and-recall

def fscore_threshold(beta=1, threshold=0.5):
    def fscore(y_true, y_pred):
        threshold_value = threshold
        beta_value = beta
        
        # Precision
        
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        
        
        # Recall
        
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        
        f_score = (1+(beta_value*beta_value)) * (precision_ratio*recall_ratio) / (beta_value*beta_value*precision_ratio + recall_ratio)
        
        return fscore_ratio
    return f_score
    
    
        