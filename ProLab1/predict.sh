#!/usr/bin/env bash

fstcompile --isymbols=chars.syms --osymbols=chars.syms pred_word.fst |
    fstcompose - ${1} |
    fstshortestpath |
    fstrmepsilon |
    fsttopsort |
    fstprint -osymbols=chars.syms |
    cut -f4 |
    grep -v "EPS" |
    head -n -1 |
    tr -d '\n'
