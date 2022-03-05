###
 # @Author: Xiang Pan
 # @Date: 2022-02-04 02:22:59
 # @LastEditTime: 2022-03-04 22:40:02
 # @LastEditors: Xiang Pan
 # @Description: 
 # @FilePath: /HW2/make_submission.sh
 # @email: xiangpan@nyu.edu
### 
rm -r ./xp2030
mkdir ./xp2030
rm -r ./xp2030.zip
cp ./build/main.pdf ./xp2030/hw2_theory.pdf
cp ./hw2_cnn.ipynb ./xp2030/
cp ./hw2_rnn.ipynb ./xp2030/
cp ./08-seq_classification.ipynb ./xp2030/
zip -r ./xp2030.zip ./xp2030/


