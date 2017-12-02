#!/usr/bin/env bash
NAME="results.csv";
echo "Compiling results from batch";
cat cs*o* >> $NAME;
echo "Cleaning up batch";
rm cs*.o*;
