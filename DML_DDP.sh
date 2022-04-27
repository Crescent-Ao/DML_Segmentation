python QAQ.py
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    DML_DDP.py
