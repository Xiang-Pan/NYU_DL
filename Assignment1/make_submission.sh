###
 # @Author: Xiang Pan
 # @Date: 2022-02-04 02:22:59
 # @LastEditTime: 2022-02-17 23:43:41
 # @LastEditors: Xiang Pan
 # @Description: 
 # @FilePath: /Assignment1/make_submission.sh
 # @email: xiangpan@nyu.edu
### 
rm -r ./xp2030
mkdir ./xp2030
cp ./mlp.py ./xp2030/
cp ./build/main.pdf ./xp2030/theory.pdf
zip -r ./xp2030.zip ./xp2030/


