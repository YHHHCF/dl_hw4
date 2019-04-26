mkdir data/
cd data/
kaggle competitions download -c 11785-s19-hw4p2
unzip dev.npy.zip
unzip test.npy.zip
unzip train_transcripts.npy.zip
unzip train.npy.zip
mkdir zip
mv *.zip zip
