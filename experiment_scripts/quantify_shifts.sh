python3 quantify_shifts.py --dataset=cifar10 --shift_sequence=corruption:label_flip:rotation --source_sample_size=6000 --target_sample_size_seq=4000:6000:4000 --target_test_size=5000
python3 quantify_shifts.py --dataset=cifar10 --shift_sequence=rotation:corruption:label_flip --source_sample_size=6000 --target_sample_size_seq=4000:6000:4000 --target_test_size=5000
python3 quantify_shifts.py --dataset=cifar10 --shift_sequence=rotation_cond:rotation_cond:rotation_cond --source_sample_size=6000 --target_sample_size_seq=4000:6000:4000 --target_test_size=5000
python3 quantify_shifts.py --dataset=cifar10 --shift_sequence=recolor:recolor:recolor --source_sample_size=6000 --target_sample_size_seq=4000:6000:4000 --target_test_size=5000
python3 quantify_shifts.py --dataset=cifar100 --shift_sequence=subpop:subpop:subpop --source_sample_size=6000 --target_sample_size_seq=4000:6000:4000 --target_test_size=5000
python3 quantify_shifts.py --dataset=portraits

python3 summarize_shift_quantities.py