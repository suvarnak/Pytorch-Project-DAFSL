

cd ../data

if [ -d "omniglot" ] 
then
    echo "Directory omniglot exists." 
else
		mkdir omniglot/
		cd omniglot
		wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
		wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip
		unzip images_background.zip
		unzip images_evaluation.zip
		cd ..
fi
if [ -d "aircraft" ] 
then
    echo "Directory aircraft exists." 
else
		mkdir aircraft
		cd aircraft/
		wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
		tar xvf fgvc-aircraft-2013b.tar.gz
		cd ..
fi

if [ -d "CUB_200_2011" ] 
then
    echo "Directory CUB_200_2011 exists." 
else
		mkdir CUB_200_2011/
		cd CUB_200_2011/
		wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
		tar xvf CUB_200_2011.tgz
		cd ..
fi

echo "downloading of datasets is complete"
