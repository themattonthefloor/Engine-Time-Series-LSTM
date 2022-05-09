# A sample inference script. 
import argparse
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

from src.tools import DataPipeline

class Inference:
    def __init__(self):
        self.dpl = DataPipeline()

    def prep_input(self, df):
        """Takes in a dataframe for a single engine and returns the
        windowed sequences (WILL NOT create the RC column/target label)"""
        assert len(df)>=100, "Engine requires a minimum of 100 cycles."
        self.dpl.ingest_inference_data(df)
        self.dpl._scale_data()
        self.dpl._perform_smoothing(alpha=0.3)
        params = {'sequence_length':100,
            'feature_cols':self.dpl.remaining_features}
        X = np.stack(list(self.dpl._data_generator(self.dpl.df,return_label=False,**params)))
        return X

    def load_model(self,path_to_h5):
        return load_model(path_to_h5)
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--data_path', default='data/turbofan.csv', type=str)
    parser.add_argument('--weights_path', default='weights/lstm.h5', type=str)  

    args = parser.parse_args()

    # Pick 1 random row from dataset
    sample_df = pd.read_csv(args.data_path)
    # Pick engine #5 since we know that it is part of the test dataset
    sample_engine = sample_df[sample_df.unit_number==5]

    inf = Inference()
    sample_X = inf.prep_input(sample_engine)


    model = inf.load_model(args.weights_path)
    prediction = model.predict(sample_X)
    print(prediction)
