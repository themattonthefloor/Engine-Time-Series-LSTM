import argparse
import pandas as pd

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.tools import DataPipeline

class TrainingPipeline:
    def __init__(self):
        self.callback = EarlyStopping(monitor='val_loss',patience=3)

    def create_model(self, input_shape, nodes_per_layer, dropout, activation):
        """Method to create a LSTM model with dropout.

        Args:
            input_shape (tuple): tuple with 2 integers, ie. (10,15)
            nodes_per_layer (list): list of nodes in each layer, ie. [64,128]
            dropout (float): The amount of dropout after each LSTM layer.
            activation (str): The activation function to use.

        Returns:
            tf.model: A tensorflow model instance.
        """
        model = Sequential()
        for idx, nodes in enumerate(nodes_per_layer):
            if idx==0:
                if idx==len(nodes_per_layer)-1:
                    model.add(LSTM(nodes,input_shape=input_shape,activation=activation))
                else:
                    model.add(LSTM(nodes,input_shape=input_shape,activation=activation, return_sequences=True))
            else:
                if idx==len(nodes_per_layer)-1:
                    model.add(LSTM(nodes,activation=activation))
                else:
                    model.add(LSTM(nodes,activation=activation, return_sequences=True))
            model.add(Dropout(dropout))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        
        return model
    

if __name__=='__main__':

    # Setting parser argument defaults to the champion parameters as chosen in EDA
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--data_path', default='data/turbofan.csv', type=str)
    parser.add_argument('--log_path', default='logs/log.csv', type=str)
    parser.add_argument('--weights_path', default='weights/lstm.h5', type=str)
    parser.add_argument('--alpha', default=0.3, type=float)
    parser.add_argument('--sequence_length', default=100, type=int)
    parser.add_argument('--nodes_per_layer', default='32:64', type=str)
    parser.add_argument('--dropout', default=0.4, type=float)
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--verbose', default=True, type=bool)

    args = parser.parse_args()
    args.nodes_per_layer = [int(x) for x in args.nodes_per_layer.split(':')]

    # Prepare the data
    dpl = DataPipeline()
    dpl.ingest_data(args.data_path)
    X_train, X_test, y_train, y_test = dpl.fit_transform(args.alpha, args.sequence_length)
    
    # Create Model
    input_shape = (X_train.shape[1],X_train.shape[2])
    tpl = TrainingPipeline()
    model = tpl.create_model(input_shape, 
        args.nodes_per_layer, 
        args.dropout, 
        args.activation)

    # Fit model using the chosen parameters
    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_test,y_test),
                        epochs=100,
                        batch_size=args.batch_size,
                        callbacks=[tpl.callback],
                        verbose=args.verbose
                        )

    # Save results to the results file
    MSE = history.history['val_loss'][-1]
    results = pd.DataFrame({'MSE':MSE,
                            'alpha':args.alpha,
                            'sequence_length':args.sequence_length,
                            'nodes_per_layer':args.nodes_per_layer,
                            'dropout':args.dropout,
                            'activation':args.activation,
                            'batch_size':args.batch_size})
    
    print(f'The MSE is {MSE:.2f}, the RMSE is {MSE**0.5:.2f}.')
    results.to_csv(args.log_path, index=False)
    model.save(args.weights_path)