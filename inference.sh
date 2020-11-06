#!/bin/bash
inputfile=""
DBfile=""
model=""
outputfile=""

while [[ $# >0 ]]
do
	key="$1"
	case $key in
		-h|--help)
		echo -e "Usage:\n"
		echo -e "inference.sh [OPTION}...<PARAM>...\n\n"
		echo -e "	-in\t ms2 files for experimental mass spectrum\n"
		echo -e "	-s\t database searching result file from Comet, should be .pin format\n"
		echo -e "	-m\t deep learning model used for filtering\n"
		echo -e "	-o\t output file stored the rescoring PSMs\n"
		exit 1
		;;
		-in)
		inputfile="$2"
		shift
		;;
		-s)
		DBfile="$2"
		shift
		;;
		-m)
		model="$2"
		shift
		;;
		-o)
		outputfile="$2"
		shift
		;;
		*)
		echo "ERROR: Unidentified user variable $key"
		exit 1
		;;
	esac
	shift
done

expEncode="test.expEncode.txt"
theoryEncode="test.theoryEncode.txt"
tempcharge="testcharge.txt"
tempidx="testidx.txt"
temppeptide="testpeptide.fasta"
theoryEncode="test.theoryEncode.txt"
tempfeature="test.feature.txt"
echo -e "Generate experimental mass spectrum group by charge\n"
python train_process.py $inputfile $expEncode
echo -e "Generate isotope distribution preprocessing files and 11 extra feature of each PSM\n"
python theory_process.py $DBfile $tempidx $tempcharge $temppeptide $tempfeature
echo -e "Generate isotope distribution for each peptide\n"
./Sipros_OpenMP -i1 $tempidx -i2 $tempcharge -i3 $temppeptide -i4 $theoryEncode
echo -e "Rescore the PSMs\n"
python Predict.py $expEncode $theoryEncode $tempfeature $outputfile $model
