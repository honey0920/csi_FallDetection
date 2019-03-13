# Fall detection for CSI data

Implement deep-learning methods CNN(Convolutional Neutral Network), GRU(Gated Recurrent
Unit) and LSTM(Long Short Term Memory) in Wi-Fi Channel State Information analysis.

My implementation is based on the projects: https://github.com/ermongroup/Wifi_Activity_Recognition, 

## Usage

### Prerequisites
1. Python 2.7
2. Python packages : numpy, pandas, matplotlib, sklearn, tensorflow >= 1.0
3. dataset : download [here](https://drive.google.com/open?id=1AvafhK9raj4CslHtGKGexHIOTJgXMCG9)



### Running

1. Run the **cross_vali_data_convert_merge.py**, which generate the training data in "input_files" folder.
2. Run the **cross_vali_lstm.py/cross_vali_gru.py/cross_vali_cnn.py**  



## References


* Yousefi S , Narui H , Dayal S , et al. A Survey on Behavior Recognition Using WiFi Channel State Information[J]. IEEE Communications Magazine, 2017, 55(10):98-104.
* Hanni Cheng, Jin Zhang, Yayu Gao and Xiaojun Hei, ”Implementing Deep Learning in Wi-Fi
Channel State Information Analysis for Fall Detection,” IEEE International Conference on Consumer
Electronics - Taiwan (ICCE-TW 2019)


