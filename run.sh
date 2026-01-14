condor_rm hhao9 

echo "Cleaning old logs and responses..."
rm -f log/*
rm -f responses/*

echo "Submitting job..."
condor_submit gpu-lab.sub
