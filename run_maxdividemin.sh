#!/usr/bin/env bash
# run maxminusmin
python main.py --env-name "UpNDownDeterministic-v4" --log-dir-pre "./logs/" --log-dir "UpNDown-maxminusmin-coef_0.01" --save-sigmas 1 --save-sigmas-every 1500 --add-rank-reg 1 --rank-reg-type "maxminusmin" --rank-reg-coef 0.01 --seed 1
