import argparse

if __name__==__main__:
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--alpha', default=0.3, type=float, required=True)
    parser.add_argument('--sequence_length', default=100, type=int, required=True)
    parser.add_argument('--nodes_per_layer', default='32:64', type=str, required=True)
    parser.add_argument('--dropout', default=0.4, type=float, required=True)
    parser.add_argument('--activation', default='tanh', type=str, required=True)
    parser.add_argument('--batch_size', default=128, type=int, required=True)

    args = parser.parse_args()


    X_train, X_test, y_train, y_test = prep_data(df,feature_removal,remaining_features,alpha,sequence_length)
    input_shape = (X_train.shape[1],X_train.shape[2])
    model = create_model(input_shape, nodes_per_layer, dropout, activation)
    callback = EarlyStopping(monitor='val_loss',patience=2)

    # Fit model using the chosen parameters
    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_test,y_test),
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[callback],
                        verbose=0
                        )

    # Save results to the HP tuning results file
    MSE = history.history['val_loss'][-1]
    results = results.append(pd.DataFrame({'MSE':MSE,
                                        'alpha':alpha,
                                        'sequence_length':sequence_length,
                                        'nodes_per_layer':[nodes_per_layer],
                                        'dropout':dropout,
                                        'activation':activation,
                                        'batch_size':batch_size}))

    results = results.sort_values(['MSE']).reset_index(drop=True)
    results.to_csv(save_DIR, index=False)