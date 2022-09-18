#!/bin/bash

set -x

if command -v python
then
    pythonCommand=python
else
    if command -v python3
    then
        pythonCommand=python3
    else
        echo "python / python3 could not be found"
        exit
    fi
fi

function runPythonBenchmark(){
  cd ${PYTHON_FOLDER}
  $pythonCommand -m venv .env
  source .env/bin/activate
  pip install --upgrade --quiet pip
  pip install --quiet -r requirements.txt
  maturin develop
  echo "Starting Python benchmarks"
  cd ${BENCH_FOLDER}/python 
  pip install --quiet -r requirements.txt
  python python_benchmark.py --resultsFile=../$1
  # exit python virtualenv
  deactivate
}

function runNodeBenchmark(){
  cd ${BENCH_FOLDER}/../node
  npm i
  npm run build-internal
  rm -rf build-ts
  npm run build
  cd ${BENCH_FOLDER}/node
  npm i
  npx tsc
  npm run bench -- --resultsFile=../$1
}

function runCSharpBenchmark(){
  cd ${BENCH_FOLDER}/csharp
  dotnet clean
  dotnet build
  dotnet run --property:Configuration=Release --resultsFile=../$1
}


script=`pwd`/${BASH_SOURCE[0]}
RELATIVE_BENCH_PATH=`dirname ${script}`
export BENCH_FOLDER=`realpath ${RELATIVE_BENCH_PATH}`
export PYTHON_FOLDER="${BENCH_FOLDER}/../python"
export BENCH_RESULTS_FOLDER="${BENCH_FOLDER}/results"
identifier=$(date +"%F")-$(date +"%H")-$(date +"%M")-$(date +"%S")
# Create results folder 
mkdir -p $BENCH_RESULTS_FOLDER
pythonResults=results/python-$identifier.json
runPythonBenchmark $pythonResults

csharpResults=results/csharp-$identifier.json
runCSharpBenchmark $csharpResults

NODE_FOLDER="${BENCH_FOLDER}/../node"
nodeResults=results/node-$identifier.json
runNodeBenchmark $nodeResults

cd ${BENCH_FOLDER}
finalCSV=results/final-$identifier.csv
$pythonCommand csv_exporter.py $pythonResults $csharpResults $nodeResults $finalCSV