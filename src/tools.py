# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.drop_features = ['setting3','sensor1','sensor5','sensor9',\
            'sensor10','sensor14','sensor16','sensor18','sensor19']
        
    def ingest_data(self,data_dir='data/turbofan.csv'):
        """Ingests the dataset from a .csv file.

        Args:
            data_dir (str, optional): Path to turbofan.csv. 
            Defaults to '../data/turbofan.csv'.
        """

        self.df = pd.read_csv(data_dir)
        self.remaining_features = [x for x in self.df.loc[:,\
            'setting1':'sensor21'].columns if x not in self.drop_features]
    
    def ingest_inference_data(self, df):
        self.df  = df.sort_values(['time'])
        self.remaining_features = [x for x in self.df.loc[:,\
            'setting1':'sensor21'].columns if x not in self.drop_features]

    def _add_remaining_cycles(self,df):
        """Appends the RC (remaining cycles) column to the dataframe"""
        
        _df = df.copy()
        total_cycles_df = _df[['unit_number','time']].groupby(['unit_number']).max()
        total_cycles_dict = dict(zip(total_cycles_df.index, total_cycles_df.time))
        _df['RC'] = _df.unit_number.map(total_cycles_dict)-_df.time
        return _df

    def _scale_data(self):
        """Performs a Standard Scaling for all columns except the label."""
        
        scale_columns = self.df.loc[:,'setting1':'sensor21'].columns
        self.scaler.fit(self.df[scale_columns])
        self.df[scale_columns] = self.scaler.transform(self.df[scale_columns])

    def _perform_smoothing(self, alpha, n_samples=0):
        """Performs a smoothing of all sensor signals."""
        
        df = self.df.copy()
        # First, calculate the exponential weighted mean of desired sensors
        sensors = [x for x in df.columns if x.startswith('sensor')]
        df[sensors] = df.groupby('unit_number')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean())
        
        # Second, drop first n_samples of each unit_nr to reduce filter delay
        def create_mask(data, samples):
            result = np.ones_like(data)
            result[0:samples] = 0
            return result
        
        mask = df.groupby('unit_number')['unit_number'].transform(create_mask, samples=n_samples).astype(bool)
        df = df[mask]
        self.df = df

    def _train_test_split_by_engine(self, verbose=False):
        """Splits the dataframe into a train set and test set by the engine unit number."""
        
        train_df = self.df[self.df.unit_number%5!=0].copy()
        test_df = self.df[self.df.unit_number%5==0].copy()
        if verbose:
            print(f'Train set engine numbers: {train_df.unit_number.unique()}')
            print(f'Test set engine numbers: {test_df.unit_number.unique()}')
        
        return train_df, test_df

    def _data_generator(self,df,sequence_length,return_label,feature_cols):
        """A sequenced window generator. Will yield a windowed frame of either target labels or input features."""
        if 'RC' in df.columns:
            _df = df.sort_values(by=['unit_number','RC'],ascending=[True,False]).reset_index(drop=True)
        else:
            _df = df.copy()
        units = _df.unit_number.unique()
        for unit in units:
            unit_df = _df[_df.unit_number==unit].reset_index(drop=True)
            for i in range(len(unit_df)-sequence_length+1):
                frame_df = unit_df.iloc[i:i+sequence_length]
                if return_label:
                    target = np.array(frame_df.iloc[-1]['RC'])
                    yield target
                else: 
                    features = np.array(frame_df[feature_cols].values)
                    yield features

    def fit_transform(self,
            alpha,
            sequence_length):
        self.df = self._add_remaining_cycles(self.df)
        self.df.drop(columns=['time'], inplace=True)
        self._scale_data()
        self._perform_smoothing(alpha)
        train_df, test_df = self._train_test_split_by_engine()

        # Drop Features
        train_df = train_df[['unit_number','RC']+self.remaining_features]
        test_df = test_df[['unit_number','RC']+self.remaining_features]

        params = {'sequence_length':sequence_length,
            'feature_cols':self.remaining_features}
        self.X_train = np.stack(list(self._data_generator(train_df,return_label=False,**params)))
        self.y_train = np.stack(list(self._data_generator(train_df,return_label=True,**params)))
        self.X_test = np.stack(list(self._data_generator(test_df,return_label=False,**params)))
        self.y_test = np.stack(list(self._data_generator(test_df,return_label=True,**params)))
        
        return self.X_train, self.X_test, self.y_train, self.y_test

if __name__ == '__main__':
    args = {'alpha':0.3,
            'sequence_length':100,
            'nodes_per_layer':[32,64],
            'dropout':0.4,
            'activation':'tanh',
            'batch_size':128}
    
    dpl = DataPipeline()
    dpl.ingest_data()
    X_train, X_test, y_train, y_test = dpl.fit_transform(args['alpha'], args['sequence_length'])
    print('X_train:',X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:',y_train.shape)
    print('y_test:',y_test.shape)

