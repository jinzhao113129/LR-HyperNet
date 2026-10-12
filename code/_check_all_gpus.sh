#!/bin/bash
for node in gpu01 gpu02 gpu04 gpu05 gpu07 gpu09 gpu10 gpu11 gpu13 gpu16 gpu19 gpu20 gpu21; do
    gres=$(scontrol show node $node 2>&1 | grep "Gres=" | head -1)
    alloc=$(scontrol show node $node 2>&1 | grep "AllocTRES" | head -1)
    echo "$node: $gres | $alloc"
done
