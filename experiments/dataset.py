# -*- coding: utf-8 -*-
# @Author: NewCoderQ
# @Date:   2018-04-03 15:46:30
# @Last Modified by:   NewCoderQ
# @Last Modified time: 2018-04-03 20:24:49

"""
    Load data from the .pkl file

    Feature name:           122
        wavelet_featrue     60
        GGCM_feature        15
        sketch_featrue      23
        GLCM_feature        24
"""

import pickle
import numpy as np


class Cataract_data():
    """A class for Cataract
    """
    def __init__(self):
        print('load cataract data...')
        self.data = np.zeros((7851, 122))
        self.label = np.zeros(7851)
        self.load_data_Cataract('../cataract_feature/name_label.pkl', 
                                '../cataract_feature/all_cataract_feature.pkl')


    def combine_feature(self, single_image_feature):
        """Combine feature from 4 sub_features

            Parameters:
                single_image_feature: single image feature

            Return:
                feature_array_1d: the 1d feature array
        """
        feature_array_1d = single_image_feature['wavelet_featrue']
        feature_array_1d = np.append(feature_array_1d, single_image_feature['GGCM_feature'])
        feature_array_1d = np.append(feature_array_1d, single_image_feature['sketch_featrue'])
        feature_array_1d = np.append(feature_array_1d, single_image_feature['GLCM_feature'])

        # print(feature_array_1d)
        # print(feature_array_1d.shape)
        # print(single_image_feature['wavelet_featrue'])
        # print(single_image_feature['GGCM_feature'])
        # print(single_image_feature['sketch_featrue'])
        # print(single_image_feature['GLCM_feature'])

        return feature_array_1d


    def load_data_Cataract(self, label_file, feature_file):
        with open(feature_file, 'rb') as in_feature:
            feature = pickle.load(in_feature, encoding='latin-1')

        with open(label_file, 'rb') as name_label:
            labels = pickle.load(name_label)
        
        for name in labels.keys():      # each image name
            index = list(labels.keys()).index(name)
            self.data[index] = self.combine_feature(feature[name])
            self.label[index] = labels[name] if labels[name] == 0 else 1
            # self.label[index] = labels[name]

        print('{} instances, {} features in all'.format(self.data.shape[0], self.data.shape[1]))
        


if __name__ == '__main__':
    cataract = Cataract_data()
    cataract.load_data_Cataract('../cataract_feature/name_label.pkl', 
                                '../cataract_feature/all_cataract_feature.pkl')


