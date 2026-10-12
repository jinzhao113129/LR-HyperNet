#!/bin/bash
for node in gpu01 gpu02 gpu04 gpu05 gpu07 gpu09 gpu16 gpu19 gpu20 gpu21; do
    echo "=== $node ==="
    scontrol show node $node 2>&1 | grep -E "Gres=|AllocTRES" | head -2
done
