#!/bin/bash

# 指定目标目录，默认为当前目录
dir=${1:-.}

# 遍历目录中所有以_prior.pkl结尾的文件
for file in "$dir"/*_prior.pkl; do
    # 检查文件是否存在，避免匹配不到文件时报错
    if [ -e "$file" ]; then
        # 使用参数替换移除_prior部分
        new_name="${file%_prior.pkl}.pkl"
        mv "$file" "$new_name"
        echo "Renamed: $file -> $new_name"
    fi
done
