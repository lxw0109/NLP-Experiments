#!/usr/bin/env python3
# coding: utf-8
# File: extract_title_content.py
# Author: lxw
# Date: 12/31/17 10:38 AM

def main():
    count = 0
    f1 = open("../../data/mnt_link/200w_news_only_title_content_seg.txt", "wb")
    # f1 = open("../../data/mnt_link/200w_news_only_title_content_seg.txt", "wb")

    with open("../../data/mnt_link/200w_news_id_title_content_seg.txt") as f:
    with open("../../data/mnt_link/200w_news_id_title_content_seg.txt") as f:
        for line in f:
            count += 1
            if count == 4:
                count = 1
                continue
            if count == 2:
                f1.write(line.encode("utf-8"))
            elif count == 3:
                f1.write(line.encode("utf-8"))
            # else:    # count == 1: continue
            #    continue
    f1.close()




if __name__ == '__main__':
    main()