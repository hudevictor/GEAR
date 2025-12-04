modeladdress="/root/shared-nvme/Llama-3.2-3B-Instruct"
#python evaluation_aqua_cot.py --model modeladdress --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method KCVT --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 20
#GEAR
 python evaluation_aqua_cot.py --model ${modeladdress} --batch_size 1 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method poweriteration --attention_number 28 --quantize_bit 2 --group_size 64 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --left 0.02 --streaming --streaming_gap 64

# python evaluation_aqua_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method GEAR-KCVT --attention_number 40 --quantize_bit 4 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --left 0.02 --streaming --streaming_gap 64

# python evaluation_aqua_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method GEARL --attention_number 40 --quantize_bit 2 --group_size 64 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --streaming --streaming_gap 64

# python evaluation_aqua_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method GEARL-KCVT --attention_number 40 --quantize_bit 4 --loop 3 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --streaming --streaming_gap 64