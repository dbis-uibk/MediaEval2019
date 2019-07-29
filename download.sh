
mkdir MER31k
cd MER31k
alias gdrive.sh = 'curl gdrive.sh | bash -s'
echo "start downloading..."
gdrive.sh 1aEpkq2SK5oySwcBlbeZCSYe6AnltJBL2
gdrive.sh 1ozCeh4rNluU0CX7nUA-z44U2aKZ6CjzT
gdrive.sh 1YDDj0O2hUuSbNKl9ggyVd3Fa79bkWu8v
echo "finish downloading, start unpacking..."
sleep 5
unzip wav_22050hz.zip
sleep 2
echo "finish unpacking, delete zip file"
rm wav_22050hz.zip
echo "all done."





