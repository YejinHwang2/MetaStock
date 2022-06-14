import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Generator
from tqdm import tqdm
from collections import defaultdict

def flatten(li: List[Any]) -> Generator:
    """flatten nested list
    ```python
    x = [[[1], 2], [[[[3]], 4, 5], 6], 7, [[8]], [9], 10]
    print(type(flatten(x)))
    # <generator object flatten at 0x00000212BF603CC8>
    print(list(flatten(x)))
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ```
    Args:
        li (List[Any]): any kinds of list
    Yields:
        Generator: flattened list generator
    """
    for ele in li:
        if isinstance(ele, list) or isinstance(ele, tuple):
            yield from flatten(ele)
        else:
            yield ele

class MetaStockDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            meta_type: str ='train', 
            data_dir: Path or str ='', 
            dtype: str ='kdd17', 
            n_train_stock: int =40, 
            n_sample: int =5,
            n_lag: int =1, 
            n_stock: int =5,
            keep_support_history: bool=False,
            show_y_index: bool=False
        ):
        """
        dataset ref: https://arxiv.org/abs/1810.09936
        In this meta learning setting, we have 3 meta-test and 1 meta-train
        vertical = stocks, horizontal = time
                train      |    test
           A               |
           B   meta-train  |   meta-test
           C               |      (1)
           ----------------|-------------
           D   meta-test   |   meta-test
           E     (2)       |      (3)

        meta-test (1) same stock, different time
        meta-test (2) different stock, same time
        meta-test (3) different stock, different time
        use `valid_date` to split the train / test set
        """
        super().__init__()
        # for debugging purpose
        self.labels_dict = {
            'fall': 0, 'rise': 1, 'unchange': 2 
        }
        self.keep_support_history = keep_support_history
        self.show_y_index = show_y_index
        # data config
        self.data_dir = Path(data_dir).resolve()
        ds_info = {
            # train: (Jan-01-2007 to Jan-01-2015)
            # val: (Jan-01-2015 to Jan-01-2016)
            # test: (Jan-01-2016 to Jan-01-2017)
            'kdd17': {
                'path': self.data_dir / 'kdd17/price_long_50',
                'date': self.data_dir / 'kdd17/trading_dates.csv',
                'train_date': '2015-01-01', 
                'val_date': '2016-01-01', 
                'test_date': '2017-01-01',
            },
            # train: (Jan-01-2014 to Aug-01-2015)
            # vali: (Aug-01-2015 to Oct-01-2015)
            # test: (Oct-01-2015 to Jan-01-2016)
            'acl18': {
                'path': self.data_dir / 'stocknet-dataset/price/raw',
                'date': self.data_dir / 'stocknet-dataset/price/trading_dates.csv',
                'train_date': '2015-08-01', 
                'val_date': '2015-10-01', 
                'test_date': '2016-01-01',
            }
        }
        ds_config = ds_info[dtype]

        self.window_sizes = [5, 10, 15, 20]
        self.n_sample = n_sample
        self.n_lag = n_lag
        self.n_stock = n_stock

        # get data
        self.data = {}
        self.candidates = {}
        ps = list((self.data_dir / ds_config['path']).glob('*.csv'))
        iterator = ps[:n_train_stock] if meta_type == 'train' else ps[n_train_stock:]
        for p in tqdm(iterator, total=len(iterator), desc='Processing data and candidates'):    
            stock_symbol = p.name.rstrip('.csv')
            df_single = self.load_single_stock(p)
            if meta_type == 'train':
                df_single = df_single.loc[df_single['date'] <= ds_config['val_date']].reset_index(drop=True)
                labels_indices = self.get_candidates(df_single)
            else:
                if meta_type == 'test1':
                    df_single = df_single.loc[df_single['date'] > ds_config['val_date']].reset_index(drop=True)
                    labels_indices = self.get_candidates(df_single)
                elif meta_type == 'test2':
                        df_single = df_single.loc[df_single['date'] <= ds_config['val_date']].reset_index(drop=True)
                        labels_indices = self.get_candidates(df_single)
                elif meta_type == 'test3':
                        df_single = df_single.loc[df_single['date'] > ds_config['val_date']].reset_index(drop=True)
                        labels_indices = self.get_candidates(df_single)
                else:
                    raise KeyError('Error argument `meta_type`, should be in (train, test1, test2, test3)')

            self.data[stock_symbol] = df_single
            self.candidates[stock_symbol] = labels_indices

    def get_candidates(self, df):
        condition = df['label'].rolling(2).apply(self.check_func).shift(-self.n_lag).fillna(0.0).astype(bool)
        labels_indices = df.index[condition].to_numpy()
        return labels_indices

    def load_single_stock(self, p: Path or str):
        def longterm_trend(x: pd.Series, k:int):
            return (x.rolling(k).sum().div(k*x) - 1) * 100

        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        if 'Unnamed' in df.columns:
            df.drop(columns=df.columns[7], inplace=True)
        if 'Original_Open' in df.columns:
            df.rename(columns={'Original_Open': 'Open', 'Open': 'Adj Open'}, inplace=True)

        # Open, High, Low
        z1 = (df.loc[:, ['Open', 'High', 'Low']].div(df['Close'], axis=0) - 1).rename(
            columns={'Open': 'open', 'High': 'high', 'Low': 'low'}) * 100
        # Close
        z2 = df[['Close']].pct_change().rename(columns={'Close': 'close'}) * 100
        # Adj Close
        z3 = df[['Adj Close']].pct_change().rename(columns={'Adj Close': 'adj_close'}) * 100

        z4 = []
        for k in [5, 10, 15, 20, 25, 30]:
            z4.append(df[['Adj Close']].apply(longterm_trend, k=k).rename(columns={'Adj Close': f'zd{k}'}))

        df_pct = pd.concat([df['Date'], z1, z2, z3] + z4, axis=1).rename(columns={'Date': 'date'})
        cols_max = df_pct.columns[df_pct.isnull().sum() == df_pct.isnull().sum().max()]
        df_pct = df_pct.loc[~df_pct[cols_max].isnull().values, :]

        # from https://arxiv.org/abs/1810.09936
        # Examples with movement percent ≥ 0.55% and ≤ −0.5% are 
        # identified as positive and negative examples, respectively
        df_pct['label'] = self.labels_dict['unchange']
        df_pct.loc[(df_pct['close'] >= 0.55), 'label'] = self.labels_dict['rise']
        df_pct.loc[(df_pct['close'] <= -0.5), 'label'] = self.labels_dict['fall']
        
        return df_pct

    def check_func(self, x):
        checks = [self.labels_dict['fall'], self.labels_dict['rise']]
        return np.isin(x.values[0], checks) and np.isin(x.values[1], checks)

    @property
    def symbols(self):
        return list(self.data.keys())

    def generate_tasks(self):
        all_tasks = defaultdict()
        for window_size in self.window_sizes:
            tasks = self.generate_tasks_per_window_size(window_size)
            all_tasks[window_size] = tasks
        return all_tasks

    def generate_tasks_per_window_size(self, window_size: int):
        # tasks: {X: (n_stock, n_sample, window_size, n_in), y: (n_stock, n_sample)}
        tasks = defaultdict(list)
        for i in range(self.n_stock):
            symbol = np.random.choice(self.symbols)
            # data: {X: (n_sample, n_in), y: (n_sample,)}
            data = self.generate_task_per_window_size_and_single_stock(symbol, window_size)
            for k, v in data.items():
                tasks[k].append(v)

        for k, v in tasks.items():
            tasks[k] = list(flatten(v))

        return tasks

    def generate_task_per_window_size_and_single_stock(self, symbol: str, window_size: int):
        df_stock = self.data[symbol]
        # condition: only continious rise or fall
        # condition = df_stock['label'].rolling(2).apply(self.check_func).shift(-self.n_lag).fillna(0.0).astype(bool)
        # labels_indices = df_stock.index[condition].to_numpy()
        # code for jumpped tags like [1(support), 0, 0, 1(query)]
        # labels_indices = df_stock.index[df_stock['label'].isin([self.labels_dict['fall'], self.labels_dict['rise']])].to_numpy()
        labels_indices = self.candidates[symbol]
        labels_candidates = labels_indices[labels_indices >= window_size]
        y_s = np.array(sorted(np.random.choice(labels_candidates, size=(self.n_sample,), replace=False)))
        y_ss = y_s-window_size
        support, support_labels = self.generate_data(df_stock, y_start=y_ss, y_end=y_s)
        
        # code for jumpped tags like [1(support), 0, 0, 1(query)]
        # y_q = labels_indices[np.arange(len(labels_indices))[np.isin(labels_indices, y_s)] + self.n_lag]
        y_q = y_s + self.n_lag
        y_qs = y_s - window_size if self.keep_support_history else y_q - window_size
        query, query_labels = self.generate_data(df_stock, y_start=y_qs, y_end=y_q)
        
        return {
            'support': support, 'support_labels': support_labels,
            'query': query, 'query_labels': query_labels
        }

    def generate_data(self, df: pd.DataFrame, y_start: np.ndarray, y_end: np.ndarray):
        # generate mini task
        inputs = []
        labels = []
        for i, j in zip(y_start, y_end):
            inputs.append(df.loc[i:j-1].to_numpy()[:, 1:-1].astype(np.float64))
            if self.show_y_index:
                labels.append(j)
            else:
                labels.append(df.loc[j].iloc[-1].astype(np.uint8))

        # inputs: (n_sample, y_end-y_start, n_in), labels: (n_sample,)
        return inputs, labels

    def map_to_tensor(self, tasks: Dict[str, Any], device: None or str=None):
        if device is None:
            device = torch.device('cpu')
        else:
            device = torch.device(device)
        tensor_tasks = {}
        for k, v in tasks.items():
            tensor = torch.LongTensor if 'labels' in k else torch.FloatTensor
            tensor_tasks[k] = tensor(np.array(v)).to(device)
        return tensor_tasks