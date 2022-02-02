#!/bin/bash

BASE_DIR='..'
DELETE_CONTENTS=('.DS_Store' '__pycache__')

for content in ${DELETE_CONTENTS[@]}
do
    echo "Delete $content"
    find "$BASE_DIR" -name "$content" -print -exec rm -r {} ';' 2>/dev/null
done

echo 'DONE.'
