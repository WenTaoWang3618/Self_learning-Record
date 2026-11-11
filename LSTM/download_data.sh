#!/usr/bin/env bash
# 下载 CB513 蛋白二级结构数据集（Zhou & Troyanskaya, ICML 2014）
# 官方镜像: https://zenodo.org/records/7764556 （原 Princeton 链接已失效）
# 注意: 文件名为 .gz 但实际是未压缩的 .npy（md5 与 Zenodo 官方记录一致），下载后直接改名即可
set -e

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR" && cd "$DATA_DIR"

URL="https://zenodo.org/records/7764556/files/cb513%2Bprofile_split1.npy.gz?download=1"
MD5="a6ca35acd2a453b0186e3fe4af23011f"

if [ -f cb513+profile_split1.npy ]; then
  echo "数据已存在: $DATA_DIR/cb513+profile_split1.npy"
  exit 0
fi

echo "下载中（约 164 MB，视网速 5-10 分钟）..."
curl -L --retry 3 -o cb513_raw.bin "$URL"

echo "校验 md5..."
echo "$MD5  cb513_raw.bin" | md5sum -c -

mv cb513_raw.bin cb513+profile_split1.npy   # 实为 .npy，直接改名
echo "完成: $DATA_DIR/cb513+profile_split1.npy"
