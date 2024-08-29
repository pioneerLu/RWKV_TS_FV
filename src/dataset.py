########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from typing import Dict, List, Sequence, Any
import pandas as pd


# class TestDataset(Dataset):
#     def __init__(self, args):
#         self.args = args
#         self.seq_len = args.ctx_len
#         self.do_normalize = args.do_normalize
#         self.prefix_len = args.prefix_len
#         self.build_test_set()

#     def build_test_set(self):
#         df = pd.read_excel(self.args.data_file, sheet_name=None)
#         keys_sorted = sorted(k for k in df)
#         data_df = df[keys_sorted[-1]]
#         X = data_df["nwp_ws100"].to_numpy()[:, np.newaxis]
#         y = data_df["fj_windSpeed"].to_numpy()[:, np.newaxis]
#         print(f"input shape: {X.shape}, target shape: {y.shape}")
#         # shift the fj_windSpeed
#         shifted_list = []
#         for i in range(1, self.args.shift_steps+1):
#             shifted_windspeed = data_df["fj_windSpeed"].shift(i).fillna(0).to_numpy()[:, np.newaxis]
#             shifted_list.append(shifted_windspeed)
#         self.shifted_y = np.concatenate(shifted_list, axis=1)
#         # split X, y to chunk 
#         X_chunks = [X[i-self.prefix_len:i+self.seq_len] for i in range(0, len(X), self.seq_len) if i != 0]
#         y_chunks = [y[i:i+self.seq_len] for i in range(0, len(y), self.seq_len) if i != 0]
#         y_shifted_chunks = [self.shifted_y[i:i+self.seq_len] for i in range(0, len(self.shifted_y), self.seq_len) if i != 0]
#         print(f"input chunks: {len(X_chunks)}, target chunks: {len(y_chunks)}")
#         self.X = X_chunks
#         self.y = y_chunks
#         self.shifted_y = y_shifted_chunks

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         if self.do_normalize:
#             input_points = (self.X[idx] - self.args.X_mean) / self.args.X_std
#         else:
#             input_points = self.X[idx]
#         targets = self.y[idx]
#         shifted_targets = self.shifted_y[idx]
#         return dict(input_points=input_points, targets=targets, shifted_targets=shifted_targets)

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, args, config):
        self.args = args
        self.seq_len = args.ctx_len
        self.do_normalize = args.do_normalize
        self.prefix_len = args.prefix_len
        self.feature_name = []
        self.target_name = config['target_col'][0]
        self.features_chosen = []
        self.X = []
        self.X_std = []
        self.X_mean = []
        self.config = config
        self.build_test_set()

    def build_test_set(self):
        df = pd.read_excel(self.args.data_file, sheet_name=None)
        df_temp = pd.read_excel(self.args.data_file)
        # 表中除了date以外可能出现的特征
        for item in df_temp.columns[1:].to_list():
            self.feature_name.append(item)
        # parser传入的特征
        for features in list(self.config.values()):
            if features is not None:
                self.features_chosen = [*self.features_chosen, *features ]
            else:
                continue

        for feature in self.features_chosen:
            if not feature in self.feature_name:
                raise NameError(f'Please ensure the feature {feature} you choose do exist in {self.args.data_file}')
        # sheet_name 为时间（精确到day）
        keys_sorted = sorted(k for k in df)
        data_df = df[keys_sorted[-1]]
    
            
        
        for feature in self.features_chosen:
            # 观测协变量需要经过zeroPadding
            if  (self.config['observed_cov_cols'] is not None) and (feature in self.config['observed_cov_cols']) :
                self.X.append(data_df[feature].shift(self.args.shift_steps+1).fillna(0).to_numpy()[:,np.newaxis].astype(np.float32))
                self.X_mean.append(data_df[feature].to_numpy()[:,np.newaxis].mean())
                self.X_std.append(data_df[feature].to_numpy()[:,np.newaxis].std())
            else:
                self.X.append(data_df[feature].to_numpy()[:,np.newaxis].astype(np.float32))
                self.X_mean.append(data_df[feature].to_numpy()[:,np.newaxis].mean())
                self.X_std.append(data_df[feature].to_numpy()[:,np.newaxis].std())
 

        self.y = data_df[self.target_name].to_numpy()[:, np.newaxis].astype(np.float32)

        # ##======target_col的历史数据，使用因果卷积：滞后1步======##
        # self.shifted_y = data_df[self.target_name].shift(self.args.shift_steps+1).fillna(0).to_numpy()[:, np.newaxis].astype(np.float32)

        
        # shift the fj_windSpeed
        shifted_list = []
        for i in range(1, self.args.shift_steps + 1):
            shifted_windspeed = data_df["POWER(MW)"].shift(i).fillna(0).to_numpy()[:, np.newaxis].astype(np.float32)
            shifted_list.append(shifted_windspeed)
        self.shifted_y = np.concatenate(shifted_list, axis=1)
        

        # split X, y to chunk
        ##=====测试集：顺序取patch样本======##
        ##------seq_len=24------##
        ##====known_cov_cols（已知协变量：取seq_len/2长度的历史数据 + seq_len/2长度的未来数据）=====##
        ##====static_cov_cols（静态协变量：取seq_len/2长度的历史数据 + seq_len/2长度的未来数据）=====##
        ##====observed_cov_cols（观测协变量：取seq_len长度的历史数据)=====##
        ##====target_col（目标变量：取seq_len长度的未来数据，变换后的目标输入同样处理）======##

        X_chunks = []
        for item in self.X:
            X_chunks.append([item[i-self.prefix_len:i+self.seq_len] for i in range(0, len(item), self.seq_len) if i != 0])
        y_chunks = [self.y[i:i + self.seq_len] for i in range(0, len(self.y) - self.seq_len, self.seq_len) if i != 0 ]
        y_shifted_chunks = [self.shifted_y[i:i + self.seq_len] for i in range(0, len(self.shifted_y) - self.seq_len + 1, self.seq_len)]

        print(f"target chunks: {len(y_chunks)}")

        for idx in range(len(X_chunks)):
            print(f"feature {idx} : {len(X_chunks[idx])}")
        self.X_chunks = X_chunks
        self.y = y_chunks
        self.shifted_y = y_shifted_chunks


    def __len__(self):
        # print("self.X_chunks[0]",len(self.X_chunks[0]))
        print("self.y",len(self.y))
        return len(self.X_chunks[0])

    def __getitem__(self, idx):
        if self.do_normalize:
            input_points = [(chunk - mean) / std for chunk, mean, std in zip(self.X_chunks, self.X_mean, self.X_std)]
            # input_points = np.concatenate(input_points, axis=1)
            targets = self.y[idx]
            shifted_targets = self.shifted_y[idx]
            return {
            **{f"X{i}": (input_points[i][idx]-self.X_mean[i])/self.X_std[i] for i in range(len(input_points))},
                "targets": targets,
                "shifted_targets": shifted_targets
                }
        

        else:
            input_points = self.X_chunks
            print(idx)
            targets = self.y[idx]
            shifted_targets = self.shifted_y[idx]
            
            return {
                **{f"X{i}": input_points[i][idx] for i in range(len(input_points))},
                    "targets": targets,
                    "shifted_targets": shifted_targets
                    }
        

class TrainDataset(Dataset):
    def __init__(self, args, config):
        self.args = args
        self.seq_len = args.ctx_len
        self.label_smoothing = args.label_smoothing
        self.do_normalize = args.do_normalize
        self.prefix_len = args.prefix_len
        self.feature_name= []
        self.target_name = config['target_col'][0]
        self.X = [] 
        self.X_std= []
        self.X_mean = []
        self.config = config
        self.build_train_set()

    def build_train_set(self):
        df = pd.read_excel(self.args.data_file, sheet_name=None)
        df_temp=pd.read_excel(self.args.data_file)
        #表中除了date以外可能出现的特征
        for item in df_temp.columns[1:].to_list():
            self.feature_name.append(item)

        self.features_chosen = []
        for features in list(self.config.values()):
            if features is not None:
                self.features_chosen = [*self.features_chosen, *features ]
            else:
                continue
        #传入的特征
        for feature in self.features_chosen:
            if not feature in self.feature_name:
                raise NameError(f'Please ensure the feature {feature} you choose do exist in {self.args.data_file}')
        #sheet_name 为时间（精确到day）
        keys_sorted = sorted(k for k in df)
        data_df_list = [df[k] for k in keys_sorted[:-1]]
        if self.label_smoothing > 0:
            data_df_list_smooth = []
            for df in data_df_list:
                df[self.target_name] = df[self.target_name].rolling(window=self.label_smoothing,center=True).median()
                df.loc[df[self.target_name].isna(), self.target_name] = 0
                df[self.target_name] = df[self.target_name].fillna(0)
                data_df_list_smooth.append(df)
            data_df = pd.concat(data_df_list_smooth)
            print(f"label smoothing with window={self.label_smoothing} applied")
        else:
            data_df = pd.concat(data_df_list)
            print(f"raw data used, no label smoothing applied")

        for feature in self.features_chosen:
            ##====known_cov_cols（已知协变量，使用con1d：不用加工)=====##

            ##====static_cov_cols（静态协变量，使用con1d：不用加工)=====##

            ##====observed_cov_cols（观测协变量，使用因果卷积：滞后1步)=====##
            if  (self.config['observed_cov_cols'] is not None) and (feature in self.config['observed_cov_cols']) :
                self.X.append(data_df[feature].shift(self.args.shift_steps+1).fillna(0).to_numpy()[:,np.newaxis].astype(np.float32))
                self.X_mean.append(data_df[feature].to_numpy()[:,np.newaxis].mean())
                self.X_std.append(data_df[feature].to_numpy()[:,np.newaxis].std())
            else:
                self.X.append(data_df[feature].to_numpy()[:,np.newaxis].astype(np.float32))
                self.X_mean.append(data_df[feature].to_numpy()[:,np.newaxis].mean())
                self.X_std.append(data_df[feature].to_numpy()[:,np.newaxis].std())
            

        self.y = data_df[self.target_name].to_numpy()[:, np.newaxis].astype(np.float32)

        # print(f"input shape: {self.X.shape}, target shape: {self.y.shape}")
        for idx,data in enumerate(self.X):
            print(f"The shape of feature{idx}: ", data.shape)
        print(f"target shape: {self.y.shape}")
        
        self.y_mean = self.y.mean()
        self.y_std = self.y.std()

        ##======target_col的历史数据，使用因果卷积：滞后1步======##
        # self.shifted_y = data_df["POWER(MW)"].shift(1).fillna(0).to_numpy()[:, np.newaxis]

        
        # shift the fj_windSpeed
        shifted_list = []
        for i in range(1, self.args.shift_steps+1):
            shifted_windspeed = data_df[self.target_name].shift(i).fillna(0).to_numpy()[:, np.newaxis]
            shifted_list.append(shifted_windspeed)
        self.shifted_y = np.concatenate(shifted_list, axis=1)
        


    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        # random sample a start index
        ##=======训练集：随机取patch样本======##

        s = random.randrange(self.seq_len, self.X[0].shape[0] - self.seq_len)

        ##------seq_len=12------##
        ##====known_cov_cols（已知协变量：取seq_len/2长度的历史数据 + seq_len/2长度的未来数据:[s-self.seq_len/2:s+self.seq_len/2]）=====##
        ##====static_cov_cols（静态协变量：取seq_len/2长度的历史数据 + seq_len/2长度的未来数据:[s-self.seq_len/2:s+self.seq_len/2]）=====##
        ##====observed_cov_cols（观测协变量：取seq_len长度的历史数据:[s-self.seq_len:s])=====##
        ##====target_col（目标变量：取seq_len长度的未来数据，变换后的目标输入同样处理:[s:s+self.seq_len]）======##

        input_points = []
        if self.do_normalize:
            for idx in range(len(self.X)):
                input_points.append((self.X[idx][s-self.prefix_len:s+self.seq_len] - self.X_mean[idx])/self.X_std[idx])
                # print(((self.X[idx][s-self.prefix_len:s+self.seq_len] - self.X_mean[idx])/self.X_std[idx]).shape)
            # input_points = (self.X[s-self.prefix_len:s+self.seq_len] - self.X_mean) / self.X_std
            targets = (self.y[s:s+self.seq_len] - self.y_mean) / self.y_std
            shifted_targets = (self.shifted_y[s:s+self.seq_len] - self.y_mean) / self.y_std
        else:
            for idx in range(len(self.X)):
                input_points.append(self.X[idx][s-self.prefix_len:s+self.seq_len])
                # print(self.X[idx][s-self.prefix_len:s+self.seq_len].shape)
            # input_points = self.X[s-self.prefix_len:s+self.seq_len] # include the previous seq_len points
            targets = self.y[s:s+self.seq_len]
            shifted_targets = self.shifted_y[s:s+self.seq_len]


        return {
                **{f"X{idx}": input_points[idx] for idx in range(len(input_points))},
                "targets": targets,
                "shifted_targets": shifted_targets
            }

    







# class TestDataset(Dataset):
#     def __init__(self, args):
#         self.args = args
#         self.seq_len = args.ctx_len
#         self.do_normalize = args.do_normalize
#         self.prefix_len = args.prefix_len
#         self.build_test_set()

#     def build_test_set(self):
#         df = pd.read_excel(self.args.data_file, sheet_name=None)
#         keys_sorted = sorted(k for k in df)
#         data_df = df[keys_sorted[-1]]
#         X = data_df["nwp_ws100"].to_numpy()[:, np.newaxis]
#         y = data_df["fj_windSpeed"].to_numpy()[:, np.newaxis]
#         print(f"input shape: {X.shape}, target shape: {y.shape}")
#         # shift the fj_windSpeed
#         shifted_list = []
#         for i in range(1, self.args.shift_steps+1):
#             shifted_windspeed = data_df["fj_windSpeed"].shift(i).fillna(0).to_numpy()[:, np.newaxis]
#             shifted_list.append(shifted_windspeed)
#         self.shifted_y = np.concatenate(shifted_list, axis=1)
#         # split X, y to chunk 
#         X_chunks = [X[i-self.prefix_len:i+self.seq_len] for i in range(0, len(X), self.seq_len) if i != 0]
#         y_chunks = [y[i:i+self.seq_len] for i in range(0, len(y), self.seq_len) if i != 0]
#         y_shifted_chunks = [self.shifted_y[i:i+self.seq_len] for i in range(0, len(self.shifted_y), self.seq_len) if i != 0]
#         print(f"input chunks: {len(X_chunks)}, target chunks: {len(y_chunks)}")
#         self.X = X_chunks
#         self.y = y_chunks
#         self.shifted_y = y_shifted_chunks

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         if self.do_normalize:
#             input_points = (self.X[idx] - self.args.X_mean) / self.args.X_std
#         else:
#             input_points = self.X[idx]
#         targets = self.y[idx]
#         shifted_targets = self.shifted_y[idx]
#         return dict(input_points=input_points, targets=targets, shifted_targets=shifted_targets)
