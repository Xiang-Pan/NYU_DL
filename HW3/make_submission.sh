###
 # @Author: Xiang Pan
 # @Date: 2022-02-04 02:22:59
 # @LastEditTime: 2022-03-04 22:47:45
 # @LastEditors: Xiang Pan
 # @Description: 
 # @FilePath: /HW3/make_submission.sh
 # @email: xiangpan@nyu.edu
### 
rm -r ./xp2030
mkdir ./xp2030
rm -r ./xp2030.zip
cp ./build/main.pdf ./xp2030/hw3_theory.pdf
cp ./hw3_practice.ipynb ./xp2030/hw3_practice.ipynb
zip -r ./xp2030.zip ./xp2030/


